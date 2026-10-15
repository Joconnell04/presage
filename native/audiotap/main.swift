// presage audiotap helper.
//
// Captures the default microphone and all system audio as two 16 kHz mono int16
// streams and writes them, framed, to stdout. No virtual audio driver, no change to
// the system's output device.
//
//   stdout frame: [1 byte stream id][4 bytes little-endian payload length][int16 LE samples]
//                 stream id 0 = mic, 1 = system. 20 ms per frame (320 samples, 640 bytes).
//   stderr:       one JSON object per line, e.g. {"event":"started","stream":"system",...}
//
// Frames never touch write(2) on an audio thread (E69b). Core Audio dispatches an IOBlock
// synchronously on the device's IO thread, so a write that blocks because Python stopped
// reading the pipe (a pipe holds well under a second of these two streams) blocks the IO
// thread, and the HAL drops the input it could not deliver. Frames go into an in-memory queue
// instead (30 s per stream), a writer thread drains it to stdout, and the only thing a long
// stall can cost is the oldest queued audio, reported as {"event":"overflow"}. A reader that
// fell behind by more than half a second is reported as {"event":"backlog"}.
//
// System audio: Core Audio process tap (macOS 14.2+). CATapDescription ->
// AudioHardwareCreateProcessTap -> AudioHardwareCreateAggregateDevice (the tap as a
// sub-tap, the real output device as the main sub-device for its clock) ->
// AudioDeviceCreateIOProcIDWithBlock. AVAudioEngine cannot be retargeted onto a
// tap-backed aggregate device, so we never try.
//
// Mic: a plain IOProc on the default input device at whatever rate and buffer size
// that device is already running. Deliberately NOT AVAudioEngine, which reconfigures
// the shared input device and can disrupt Zoom/Teams mid-call.

import Foundation
import CoreAudio
import AVFoundation

let STREAM_MIC: UInt8 = 0
let STREAM_SYSTEM: UInt8 = 1
let OUT_RATE: Double = 16000
let FRAME_SAMPLES = 320          // 20 ms at 16 kHz

// ---------------------------------------------------------------- output plumbing

var stderrLock = pthread_mutex_t()
pthread_mutex_init(&stderrLock, nil)

/// false when the reader has gone away (EPIPE): nobody will ever read these frames.
@discardableResult
func writeAll(_ fd: Int32, _ bytes: UnsafeRawPointer, _ count: Int) -> Bool {
    var off = 0
    while off < count {
        let n = write(fd, bytes.advanced(by: off), count - off)
        if n <= 0 {
            if errno == EINTR { continue }
            return errno != EPIPE
        }
        off += n
    }
    return true
}

/// The queue between the audio threads and stdout. push() only takes a lock that the writer
/// holds for a swap, never across a write, so an audio thread cannot be made to wait on Python.
final class FrameOut {
    private var lock = pthread_mutex_t()
    private var cond = pthread_cond_t()
    private var pending: [[UInt8]] = []
    private let capacity: Int
    private var droppedMic = 0
    private var droppedSystem = 0
    private var peakPending = 0
    private var writing = false

    init(seconds: Double) {
        capacity = max(100, Int(seconds * OUT_RATE / Double(FRAME_SAMPLES)) * 2)
        pthread_mutex_init(&lock, nil)
        pthread_cond_init(&cond, nil)
        pending.reserveCapacity(256)
    }

    func push(_ frame: [UInt8]) {
        pthread_mutex_lock(&lock)
        if pending.count >= capacity {
            // Python has not read for the whole queue's length: lose the oldest, keep the newest.
            if pending[0][0] == STREAM_MIC { droppedMic += 1 } else { droppedSystem += 1 }
            pending.removeFirst()
        }
        pending.append(frame)
        if pending.count > peakPending { peakPending = pending.count }
        pthread_cond_signal(&cond)
        pthread_mutex_unlock(&lock)
    }

    /// The writer thread. Takes everything queued in one swap and writes it with the lock free.
    func run() {
        var batch: [[UInt8]] = []
        batch.reserveCapacity(256)
        while true {
            pthread_mutex_lock(&lock)
            while pending.isEmpty { pthread_cond_wait(&cond, &lock) }
            swap(&batch, &pending)
            writing = true
            pthread_mutex_unlock(&lock)
            for f in batch {
                let ok = f.withUnsafeBytes { writeAll(1, $0.baseAddress!, $0.count) }
                if !ok {
                    // The parent closed our stdout without stopping us. An orphaned helper
                    // would hold the microphone open for nobody; leave instead.
                    emit(["event": "stopped", "reason": "stdout_closed"])
                    exit(0)
                }
            }
            batch.removeAll(keepingCapacity: true)
            pthread_mutex_lock(&lock)
            writing = false
            pthread_mutex_unlock(&lock)
        }
    }

    /// Wait (up to `timeout` s) until everything queued has been written. Used before a
    /// planned exit so --seconds runs do not lose their last frames.
    func drain(timeout: Double) {
        let end = Date().addingTimeInterval(timeout)
        while Date() < end {
            pthread_mutex_lock(&lock)
            let idle = pending.isEmpty && !writing
            pthread_mutex_unlock(&lock)
            if idle { return }
            usleep(5000)
        }
    }

    /// (frames dropped per stream since the last call, most frames queued at once since the last call)
    func takeCounters() -> (mic: Int, system: Int, peak: Int) {
        pthread_mutex_lock(&lock)
        let r = (droppedMic, droppedSystem, peakPending)
        droppedMic = 0
        droppedSystem = 0
        peakPending = pending.count
        pthread_mutex_unlock(&lock)
        return r
    }
}

let frameOut = FrameOut(seconds: 30)
let frameWriter = Thread { frameOut.run() }
frameWriter.name = "presage.stdout"
frameWriter.qualityOfService = .userInitiated
frameWriter.start()

func emitFrame(_ stream: UInt8, _ samples: UnsafePointer<Int16>, _ count: Int) {
    let payload = count * 2
    var buf = [UInt8](repeating: 0, count: 5 + payload)
    buf[0] = stream
    buf[1] = UInt8(payload & 0xff)
    buf[2] = UInt8((payload >> 8) & 0xff)
    buf[3] = UInt8((payload >> 16) & 0xff)
    buf[4] = UInt8((payload >> 24) & 0xff)
    samples.withMemoryRebound(to: UInt8.self, capacity: payload) { src in
        _ = buf.withUnsafeMutableBufferPointer { memcpy($0.baseAddress! + 5, src, payload) }
    }
    frameOut.push(buf)
}

func emit(_ obj: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) else { return }
    var line = [UInt8](data)
    line.append(0x0a)
    pthread_mutex_lock(&stderrLock)
    _ = line.withUnsafeBytes { writeAll(2, $0.baseAddress!, $0.count) }
    pthread_mutex_unlock(&stderrLock)
}

func fourCC(_ s: OSStatus) -> String? {
    let v = UInt32(bitPattern: s)
    let b = [UInt8((v >> 24) & 0xff), UInt8((v >> 16) & 0xff), UInt8((v >> 8) & 0xff), UInt8(v & 0xff)]
    guard b.allSatisfy({ $0 >= 32 && $0 < 127 }) else { return nil }
    return String(bytes: b, encoding: .ascii)
}

func emitError(_ where_: String, _ status: OSStatus, _ extra: [String: Any] = [:]) {
    var o: [String: Any] = ["event": "error", "where": where_, "status": Int(status)]
    if let s = fourCC(status) { o["code"] = s }
    for (k, v) in extra { o[k] = v }
    emit(o)
}

// ------------------------------------------------------- mono float -> 16 kHz int16

/// Counters a stream's watchdog reads. Written from an audio thread, read from main;
/// a torn read only delays a diagnostic by a tick, so no lock.
final class StreamStats {
    var callbacks = 0
    var framesOut = 0
    var peak: Int16 = 0
}

/// Resamples mono float32 at the device's rate to 16 kHz mono int16 and emits 20 ms
/// frames. One per stream; AVAudioConverter carries the filter phase across calls, so
/// there is no per-buffer rounding drift.
final class MonoTo16k {
    private let stream: UInt8
    private let converter: AVAudioConverter
    private let srcFormat: AVAudioFormat
    private let dstFormat: AVAudioFormat
    private let ratio: Double
    private var carry = [Int16]()
    let stats: StreamStats

    init?(stream: UInt8, srcRate: Double, stats: StreamStats) {
        guard srcRate > 0,
              let src = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: srcRate,
                                      channels: 1, interleaved: false),
              let dst = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: OUT_RATE,
                                      channels: 1, interleaved: true),
              let conv = AVAudioConverter(from: src, to: dst)
        else { return nil }
        conv.sampleRateConverterQuality = AVAudioQuality.medium.rawValue
        self.stream = stream
        self.srcFormat = src
        self.dstFormat = dst
        self.converter = conv
        self.ratio = OUT_RATE / srcRate
        self.stats = stats
        carry.reserveCapacity(FRAME_SAMPLES * 4)
    }

    func push(_ mono: UnsafePointer<Float>, _ frames: Int) {
        guard frames > 0 else { return }
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: AVAudioFrameCount(frames)) else { return }
        inBuf.frameLength = AVAudioFrameCount(frames)
        memcpy(inBuf.floatChannelData![0], mono, frames * MemoryLayout<Float>.size)

        let cap = AVAudioFrameCount(Double(frames) * ratio + 64)
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: cap) else { return }

        var supplied = false
        var err: NSError?
        let status = converter.convert(to: outBuf, error: &err) { _, outStatus in
            if supplied { outStatus.pointee = .noDataNow; return nil }
            supplied = true
            outStatus.pointee = .haveData
            return inBuf
        }
        if status == .error {
            emitError("AVAudioConverter.convert", OSStatus(err?.code ?? 0),
                      ["stream": stream == STREAM_MIC ? "mic" : "system"])
            return
        }
        let n = Int(outBuf.frameLength)
        guard n > 0, let src = outBuf.int16ChannelData?[0] else { return }
        var p = stats.peak
        for i in 0..<n {
            let v = src[i]
            let a = v == Int16.min ? Int16.max : abs(v)
            if a > p { p = a }
            carry.append(v)
        }
        stats.peak = p
        stats.framesOut += n

        while carry.count >= FRAME_SAMPLES {
            carry.withUnsafeBufferPointer { emitFrame(stream, $0.baseAddress!, FRAME_SAMPLES) }
            carry.removeFirst(FRAME_SAMPLES)
        }
    }
}

/// Averages one AudioBuffer's channels down to mono. Core Audio IOProcs always hand
/// us packed float32.
func downmix(_ abl: UnsafePointer<AudioBufferList>, bufferIndex: Int, into scratch: inout [Float]) -> Int {
    let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: abl))
    guard list.count > bufferIndex else { return 0 }
    let buf = list[bufferIndex]
    guard let raw = buf.mData else { return 0 }
    let nch = max(1, Int(buf.mNumberChannels))
    let frames = Int(buf.mDataByteSize) / (4 * nch)
    guard frames > 0 else { return 0 }
    if scratch.count < frames { scratch = [Float](repeating: 0, count: frames * 2) }
    let f = raw.assumingMemoryBound(to: Float.self)
    if nch == 1 {
        _ = scratch.withUnsafeMutableBufferPointer { memcpy($0.baseAddress!, f, frames * 4) }
    } else {
        let inv = 1.0 / Float(nch)
        scratch.withUnsafeMutableBufferPointer { d in
            for i in 0..<frames {
                var acc: Float = 0
                for c in 0..<nch { acc += f[i * nch + c] }
                d[i] = acc * inv
            }
        }
    }
    return frames
}

// ---------------------------------------------------------------- device helpers

func defaultDeviceID(output: Bool) -> AudioObjectID {
    var addr = AudioObjectPropertyAddress(
        mSelector: output ? kAudioHardwarePropertyDefaultOutputDevice : kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var id = AudioObjectID(kAudioObjectUnknown)
    var sz = UInt32(MemoryLayout<AudioObjectID>.size)
    let st = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &sz, &id)
    if st != noErr { emitError("kAudioHardwarePropertyDefault\(output ? "Output" : "Input")Device", st) }
    return id
}

func stringProperty(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String {
    var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var cf: CFString? = nil
    var sz = UInt32(MemoryLayout<CFString?>.size)
    let st = withUnsafeMutablePointer(to: &cf) { AudioObjectGetPropertyData(id, &addr, 0, nil, &sz, $0) }
    guard st == noErr, let s = cf else { return "" }
    return s as String
}

func deviceName(_ id: AudioObjectID) -> String { stringProperty(id, kAudioObjectPropertyName) }
func deviceUID(_ id: AudioObjectID) -> String { stringProperty(id, kAudioDevicePropertyDeviceUID) }

func uint32Property(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector,
                    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> UInt32? {
    var addr = AudioObjectPropertyAddress(mSelector: selector, mScope: scope,
                                          mElement: kAudioObjectPropertyElementMain)
    var v: UInt32 = 0
    var sz = UInt32(MemoryLayout<UInt32>.size)
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &sz, &v) == noErr else { return nil }
    return v
}

/// One of builtin / bluetooth / usb / virtual / aggregate / other, from
/// kAudioDevicePropertyTransportType. The UI warns on "bluetooth" inputs: those are the
/// ones that run at 24 kHz and can go digitally silent (see docs/research-audio.md).
func transportType(_ id: AudioObjectID) -> String {
    guard let t = uint32Property(id, kAudioDevicePropertyTransportType) else { return "other" }
    switch t {
    case kAudioDeviceTransportTypeBuiltIn: return "builtin"
    case kAudioDeviceTransportTypeBluetooth, kAudioDeviceTransportTypeBluetoothLE: return "bluetooth"
    case kAudioDeviceTransportTypeUSB: return "usb"
    case kAudioDeviceTransportTypeVirtual: return "virtual"
    case kAudioDeviceTransportTypeAggregate: return "aggregate"
    default: return "other"
    }
}

func deviceIsRunningSomewhere(_ id: AudioObjectID) -> Bool {
    (uint32Property(id, kAudioDevicePropertyDeviceIsRunningSomewhere) ?? 0) != 0
}

func deviceIsAlive(_ id: AudioObjectID) -> Bool {
    (uint32Property(id, kAudioDevicePropertyDeviceIsAlive) ?? 0) != 0
}

/// 1 muted, 0 not muted, -1 the device has no input mute control (AirPods do not).
func inputMuted(_ id: AudioObjectID) -> Int {
    guard let v = uint32Property(id, kAudioDevicePropertyMute, scope: kAudioObjectPropertyScopeInput)
    else { return -1 }
    return v != 0 ? 1 : 0
}

/// The input volume slider, 0-1, or nil when the device has none. A quiet mic is often just
/// this: AirPods were measured at 0.24 (docs/research-audio.md). Reported, never changed.
func inputVolume(_ id: AudioObjectID) -> Double? {
    for element in [kAudioObjectPropertyElementMain, 1] {
        var addr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyVolumeScalar,
                                              mScope: kAudioObjectPropertyScopeInput,
                                              mElement: element)
        var v: Float32 = 0
        var sz = UInt32(MemoryLayout<Float32>.size)
        if AudioObjectHasProperty(id, &addr),
           AudioObjectGetPropertyData(id, &addr, 0, nil, &sz, &v) == noErr {
            return Double((v * 1000).rounded() / 1000)
        }
    }
    return nil
}

/// The Control Center "Mic Mode" in force (macOS 12+): standard, voice_isolation or
/// wide_spectrum. Voice Isolation hollows out the gaps between words, which is worth knowing
/// when a recording is being read back. Reported, never changed: only the user can change it.
func micMode() -> String {
    switch AVCaptureDevice.activeMicrophoneMode {
    case .standard: return "standard"
    case .voiceIsolation: return "voice_isolation"
    case .wideSpectrum: return "wide_spectrum"
    @unknown default: return "other"
    }
}

func inputChannelCount(_ id: AudioObjectID) -> Int { streamLayout(id, input: true).reduce(0, +) }

/// The device with this UID, or nil. Core Audio's translate property answers noErr with
/// kAudioObjectUnknown for a UID that does not exist, so a scan backs it up.
func deviceForUID(_ uid: String) -> AudioObjectID? {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyTranslateUIDToDevice,
                                          mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var cfuid = uid as CFString
    var out = AudioObjectID(kAudioObjectUnknown)
    var sz = UInt32(MemoryLayout<AudioObjectID>.size)
    let st = withUnsafeMutablePointer(to: &cfuid) { p -> OSStatus in
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr,
                                   UInt32(MemoryLayout<CFString>.size), p, &sz, &out)
    }
    if st == noErr, out != AudioObjectID(kAudioObjectUnknown) { return out }
    for id in allDeviceIDs() where deviceUID(id) == uid { return id }
    return nil
}

/// An iPhone acting as a Continuity Capture mic. A real microphone, but reaching for
/// someone's phone mid-call is a surprise, so it sorts last.
func isContinuityDevice(_ id: AudioObjectID) -> Bool {
    guard let t = uint32Property(id, kAudioDevicePropertyTransportType) else { return false }
    return t == kAudioDeviceTransportTypeContinuityCaptureWired
        || t == kAudioDeviceTransportTypeContinuityCaptureWireless
}

/// The mic to fall back to when the chosen one goes digitally silent: the built-in
/// microphone, else the best other non-Bluetooth input with channels (built-in, then
/// USB, then anything else, with an iPhone last). Bluetooth is skipped because it is
/// what we are running away from; virtual loopbacks (ZoomAudioDevice, Microsoft Teams
/// Audio) are skipped because they would be just as silent.
func fallbackInputDevice(excluding: AudioObjectID) -> AudioObjectID? {
    if let d = deviceForUID("BuiltInMicrophoneDevice"), d != excluding, inputChannelCount(d) > 0 {
        return d
    }
    func rank(_ id: AudioObjectID) -> Int {
        if isContinuityDevice(id) { return 3 }
        switch transportType(id) {
        case "builtin": return 0
        case "usb": return 1
        default: return 2
        }
    }
    return allDeviceIDs()
        .filter {
            $0 != excluding && inputChannelCount($0) > 0
                && !["bluetooth", "virtual", "aggregate"].contains(transportType($0))
        }
        .min { rank($0) < rank($1) }
}

/// Whether this process may record at all. A denied mic opens and delivers zeros, and
/// no amount of device hopping fixes that, so the recovery path checks first.
func micPermissionGranted() -> Bool {
    AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
}

/// The format an IOProc will actually hand us for this device's first input stream.
/// The device's nominal rate can disagree: AirPods report 48 kHz until something opens
/// the mic, then the input device drops to 24 kHz.
func inputStreamFormat(_ id: AudioObjectID) -> AudioStreamBasicDescription? {
    var sAddr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams,
                                           mScope: kAudioObjectPropertyScopeInput,
                                           mElement: kAudioObjectPropertyElementMain)
    var sz: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &sAddr, 0, nil, &sz) == noErr, sz > 0 else { return nil }
    var streams = [AudioObjectID](repeating: 0, count: Int(sz) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(id, &sAddr, 0, nil, &sz, &streams) == noErr,
          let first = streams.first else { return nil }
    var fAddr = AudioObjectPropertyAddress(mSelector: kAudioStreamPropertyVirtualFormat,
                                           mScope: kAudioObjectPropertyScopeGlobal,
                                           mElement: kAudioObjectPropertyElementMain)
    var asbd = AudioStreamBasicDescription()
    var fsz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    guard AudioObjectGetPropertyData(first, &fAddr, 0, nil, &fsz, &asbd) == noErr else { return nil }
    return asbd
}

func nominalSampleRate(_ id: AudioObjectID) -> Double {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyNominalSampleRate,
                                          mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var rate: Double = 0
    var sz = UInt32(MemoryLayout<Double>.size)
    let st = AudioObjectGetPropertyData(id, &addr, 0, nil, &sz, &rate)
    return st == noErr ? rate : 0
}

func streamLayout(_ id: AudioObjectID, input: Bool) -> [Int] {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamConfiguration,
        mScope: input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput,
        mElement: kAudioObjectPropertyElementMain)
    var sz: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &sz) == noErr, sz > 0 else { return [] }
    let mem = UnsafeMutableRawPointer.allocate(byteCount: Int(sz), alignment: 8)
    defer { mem.deallocate() }
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &sz, mem) == noErr else { return [] }
    return UnsafeMutableAudioBufferListPointer(mem.assumingMemoryBound(to: AudioBufferList.self))
        .map { Int($0.mNumberChannels) }
}

func allDeviceIDs() -> [AudioObjectID] {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                          mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var sz: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &sz) == noErr else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(sz) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &sz, &ids) == noErr else { return [] }
    return ids
}

/// Our own process object, so the tap can leave our own output out of the mix.
func ownProcessObjectID() -> AudioObjectID? {
    var pid = getpid()
    var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
                                          mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var obj = AudioObjectID(kAudioObjectUnknown)
    var sz = UInt32(MemoryLayout<AudioObjectID>.size)
    let st = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr,
                                        UInt32(MemoryLayout<pid_t>.size), &pid, &sz, &obj)
    return (st == noErr && obj != kAudioObjectUnknown) ? obj : nil
}

// ---------------------------------------------------------------- system capture

final class SystemCapture {
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var conv: MonoTo16k?
    private var scratch = [Float](repeating: 0, count: 8192)
    private let queue = DispatchQueue(label: "presage.tap.system", qos: .userInteractive)
    let stats = StreamStats()
    private(set) var builtRate: Double = 0

    /// Swaps in a resampler for a new source rate without touching the device. The
    /// swap runs on the IOProc's own queue, so it cannot race the audio callback.
    func retune(to rate: Double) {
        guard rate > 0, abs(rate - builtRate) > 1, conv != nil else { return }
        guard let c = MonoTo16k(stream: STREAM_SYSTEM, srcRate: rate, stats: stats) else { return }
        let old = builtRate
        builtRate = rate
        queue.async { self.conv = c }
        emit(["event": "retuned", "stream": "system", "from": old, "to": rate])
    }

    /// The tap's current format, which follows the output device's rate.
    func currentRate() -> Double {
        guard tapID != kAudioObjectUnknown else { return 0 }
        var addr = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
                                              mScope: kAudioObjectPropertyScopeGlobal,
                                              mElement: kAudioObjectPropertyElementMain)
        var asbd = AudioStreamBasicDescription()
        var sz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        guard AudioObjectGetPropertyData(tapID, &addr, 0, nil, &sz, &asbd) == noErr else { return 0 }
        return asbd.mSampleRate
    }

    func start() -> Bool {
        let outDev = defaultDeviceID(output: true)
        guard outDev != kAudioObjectUnknown else { emitError("defaultOutputDevice", -1); return false }
        let outUID = deviceUID(outDev)
        let uuid = UUID()

        // 1. a mono mixdown of every process except ours.
        let desc: CATapDescription
        if let me = ownProcessObjectID() {
            desc = CATapDescription(monoGlobalTapButExcludeProcesses: [me])
        } else {
            desc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
        }
        desc.uuid = uuid
        desc.name = "presage-system-tap"
        desc.isPrivate = true
        desc.muteBehavior = .unmuted

        var st = AudioHardwareCreateProcessTap(desc, &tapID)
        if st != noErr || tapID == kAudioObjectUnknown {
            emitError("AudioHardwareCreateProcessTap", st)
            return false
        }

        // 2. the tap's own stream format.
        var fmtAddr = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var asbd = AudioStreamBasicDescription()
        var fsz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        st = AudioObjectGetPropertyData(tapID, &fmtAddr, 0, nil, &fsz, &asbd)
        if st != noErr { emitError("kAudioTapPropertyFormat", st); stop(); return false }
        let tapChannels = Int(asbd.mChannelsPerFrame)

        // 3. an aggregate that owns the tap. Only the tap goes in; the real output
        //    device is named as main sub-device for its clock but is NOT listed as a
        //    sub-device, so a device that also has inputs cannot add its own channels
        //    to our input buffer list.
        let composition: [String: Any] = [
            kAudioAggregateDeviceNameKey: "presage system tap",
            kAudioAggregateDeviceUIDKey: "presage-agg-\(uuid.uuidString)",
            kAudioAggregateDeviceMainSubDeviceKey: outUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapDriftCompensationKey: true,
                kAudioSubTapUIDKey: uuid.uuidString,
            ]],
        ]
        st = AudioHardwareCreateAggregateDevice(composition as CFDictionary, &aggID)
        if st != noErr || aggID == kAudioObjectUnknown {
            emitError("AudioHardwareCreateAggregateDevice", st)
            stop(); return false
        }

        // 4. which input buffer is the tap's.
        let layout = streamLayout(aggID, input: true)
        var idx = max(0, layout.count - 1)
        if let i = layout.lastIndex(of: tapChannels) { idx = i }

        builtRate = asbd.mSampleRate
        guard let c = MonoTo16k(stream: STREAM_SYSTEM, srcRate: asbd.mSampleRate, stats: stats) else {
            emitError("AVAudioConverter", -1, ["stream": "system", "rate": asbd.mSampleRate])
            stop(); return false
        }
        conv = c

        st = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { [weak self] _, inInput, _, _, _ in
            guard let self = self, let conv = self.conv else { return }
            self.stats.callbacks += 1
            let n = downmix(inInput, bufferIndex: idx, into: &self.scratch)
            if n > 0 { self.scratch.withUnsafeBufferPointer { conv.push($0.baseAddress!, n) } }
        }
        if st != noErr || procID == nil { emitError("AudioDeviceCreateIOProcIDWithBlock", st); stop(); return false }

        st = AudioDeviceStart(aggID, procID)
        if st != noErr { emitError("AudioDeviceStart", st, ["stream": "system"]); stop(); return false }

        emit(["event": "started", "stream": "system", "device": deviceName(outDev), "uid": outUID,
              "src_rate": asbd.mSampleRate, "src_channels": tapChannels,
              "agg_input_layout": layout, "tap_buffer_index": idx])
        return true
    }

    func stop() {
        if aggID != kAudioObjectUnknown {
            if let p = procID {
                AudioDeviceStop(aggID, p)
                AudioDeviceDestroyIOProcID(aggID, p)
                procID = nil
            }
            AudioHardwareDestroyAggregateDevice(aggID)
            aggID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
        conv = nil
    }
}

// ---------------------------------------------------------------- mic capture

final class MicCapture {
    private(set) var devID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var conv: MonoTo16k?
    private var scratch = [Float](repeating: 0, count: 8192)
    private let queue = DispatchQueue(label: "presage.mic", qos: .userInteractive)
    let stats = StreamStats()
    private(set) var builtRate: Double = 0

    /// The device this capture is on, filled in by start().
    private(set) var uid = ""
    private(set) var name = ""

    /// --input-uid: the device the user asked for, instead of the default input.
    var preferredUID: String?
    /// Set by the recovery path to pin the capture to a specific device (the built-in
    /// mic) regardless of what the default input is. Cleared on any real device change.
    var overrideDeviceID: AudioObjectID?
    private(set) var isFallback = false

    /// --debug-mute-mic-after: zeroes this stream's samples to fake a mic that has gone
    /// digitally silent. Armed for one device UID only, so the fallback device still
    /// delivers audio and the recovery path can be tested end to end.
    var debugMuteUID: String?
    private var debugMuted = false

    /// Swaps in a resampler for a new source rate without stopping the device. Doing
    /// this by stop/start instead would never settle on a Bluetooth headset: closing
    /// the IOProc lets the device go back to 48 kHz, and opening it drops it to 24 kHz
    /// again, so every rebuild would re-create the same mismatch. The swap runs on the
    /// IOProc's own queue, so it cannot race the audio callback.
    func retune(to rate: Double) {
        guard rate > 0, abs(rate - builtRate) > 1, conv != nil else { return }
        guard let c = MonoTo16k(stream: STREAM_MIC, srcRate: rate, stats: stats) else { return }
        let old = builtRate
        builtRate = rate
        queue.async { self.conv = c }
        emit(["event": "retuned", "stream": "mic", "from": old, "to": rate])
    }

    /// The rate the device we are open on is running now, which is not always the rate
    /// it was on when we built the resampler. Reads the device we actually opened, not
    /// the default input: with --input-uid or after a fallback those are different
    /// devices, and retuning to the wrong one's rate halves or doubles the audio.
    func currentRate() -> Double {
        guard devID != kAudioObjectUnknown else { return 0 }
        return inputStreamFormat(devID)?.mSampleRate ?? nominalSampleRate(devID)
    }

    /// Which device to open: an explicit override (the fallback mic), else --input-uid,
    /// else the default input.
    private func resolveDevice() -> AudioObjectID {
        if let o = overrideDeviceID, o != kAudioObjectUnknown, inputChannelCount(o) > 0 { return o }
        overrideDeviceID = nil
        isFallback = false
        if let want = preferredUID {
            if let d = deviceForUID(want), inputChannelCount(d) > 0 { return d }
            let fallback = defaultDeviceID(output: false)
            emit(["event": "warning", "stream": "mic", "reason": "input_uid_not_found",
                  "uid": want, "using": deviceName(fallback), "using_uid": deviceUID(fallback)])
        }
        return defaultDeviceID(output: false)
    }

    func start() -> Bool {
        devID = resolveDevice()
        guard devID != kAudioObjectUnknown else {
            emitError("defaultInputDevice", -1, ["stream": "mic"]); return false
        }
        uid = deviceUID(devID)
        name = deviceName(devID)
        debugMuted = debugMuteUID != nil && debugMuteUID == uid
        let layout = streamLayout(devID, input: true)
        guard !layout.isEmpty else {
            emitError("kAudioDevicePropertyStreamConfiguration", -1,
                      ["stream": "mic", "device": name, "uid": uid,
                       "reason": "input device has no input channels"])
            devID = AudioObjectID(kAudioObjectUnknown)
            return false
        }
        let rate = inputStreamFormat(devID)?.mSampleRate ?? nominalSampleRate(devID)
        builtRate = rate
        guard let c = MonoTo16k(stream: STREAM_MIC, srcRate: rate, stats: stats) else {
            emitError("AVAudioConverter", -1, ["stream": "mic", "rate": rate]); return false
        }
        conv = c

        var st = AudioDeviceCreateIOProcIDWithBlock(&procID, devID, queue) { [weak self] _, inInput, _, _, _ in
            guard let self = self, let conv = self.conv else { return }
            self.stats.callbacks += 1
            let n = downmix(inInput, bufferIndex: 0, into: &self.scratch)
            guard n > 0 else { return }
            if self.debugMuted {
                _ = self.scratch.withUnsafeMutableBufferPointer { memset($0.baseAddress!, 0, n * 4) }
            }
            self.scratch.withUnsafeBufferPointer { conv.push($0.baseAddress!, n) }
        }
        if st != noErr || procID == nil {
            emitError("AudioDeviceCreateIOProcIDWithBlock", st, ["stream": "mic"]); stop(); return false
        }
        st = AudioDeviceStart(devID, procID)
        if st != noErr { emitError("AudioDeviceStart", st, ["stream": "mic"]); stop(); return false }

        var started: [String: Any] = ["event": "started", "stream": "mic", "device": name, "uid": uid,
                                      "src_rate": rate, "src_channels": layout.first ?? 0,
                                      "input_layout": layout, "transport": transportType(devID)]
        if let v = inputVolume(devID) { started["input_volume"] = v }
        started["mic_mode"] = micMode()
        if isFallback { started["fallback"] = true }
        if debugMuted { started["debug_muted"] = true }
        emit(started)

        // Opening this IOProc is itself what makes a Bluetooth headset switch to its
        // headset profile and drop the input device to 24 kHz, so the rate we read a
        // moment ago is already stale. Re-read now, and again shortly, before the 1 s
        // poll would catch it.
        retune(to: currentRate())
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
            guard let self = self else { return }
            self.retune(to: self.currentRate())
        }
        return true
    }

    /// Recovery step 1: close and re-open the IOProc on the same device. Pins the
    /// device so that a default input that has meanwhile moved does not send us
    /// somewhere else mid-recovery.
    func reopenSameDevice() -> Bool {
        let d = devID
        stop()
        if d != kAudioObjectUnknown { overrideDeviceID = d }
        return start()
    }

    /// Recovery step 2: move to another device (the built-in mic) and stay there.
    func switchTo(_ device: AudioObjectID, fallback: Bool) -> Bool {
        stop()
        overrideDeviceID = device
        isFallback = fallback
        let ok = start()
        if !ok { isFallback = false }
        return ok
    }

    /// Testing only: start zeroing this device's samples from now on. A reopen or a
    /// move to another device re-evaluates it by UID, so the fallback mic is unaffected.
    func armDebugMute() {
        debugMuteUID = uid
        debugMuted = true
    }

    /// Forget the pin, so the next start() honours --input-uid or the default input
    /// again. Called when a real device change makes the fallback obsolete.
    func clearPin() {
        overrideDeviceID = nil
        isFallback = false
    }

    func stop() {
        if devID != kAudioObjectUnknown, let p = procID {
            AudioDeviceStop(devID, p)
            AudioDeviceDestroyIOProcID(devID, p)
            procID = nil
        }
        conv = nil
    }
}

// ---------------------------------------------------------------- main

var wantMic = true
var wantSystem = true
var seconds: Double = 0
var listOnly = false
var inputUID: String?
var debugMuteMicAfter: Double = 0
var debugSynth = false

var argi = 1
let args = CommandLine.arguments
while argi < args.count {
    switch args[argi] {
    case "--list-devices": listOnly = true
    case "--mic-only": wantSystem = false
    case "--system-only": wantMic = false
    case "--seconds":
        argi += 1
        if argi < args.count { seconds = Double(args[argi]) ?? 0 }
    case "--input-uid":
        argi += 1
        if argi < args.count { inputUID = args[argi] }
    case "--debug-synth":
        // Testing hook (E69b): open no device at all; a 440 Hz tone at -30 dBFS on the mic
        // stream, produced on the clock the way hardware is, whether or not anyone reads.
        debugSynth = true
    case "--debug-mute-mic-after":
        // Testing hook, documented in docs/research-audio.md: zero the mic samples
        // after N seconds to fake a device that has gone digitally silent.
        argi += 1
        if argi < args.count { debugMuteMicAfter = Double(args[argi]) ?? 0 }
    default:
        emit(["event": "error", "where": "args", "status": -1, "arg": args[argi]])
    }
    argi += 1
}

if listOnly {
    let defOut = defaultDeviceID(output: true)
    let defIn = defaultDeviceID(output: false)
    for id in allDeviceIDs() {
        emit(["event": "device", "id": Int(id), "name": deviceName(id), "uid": deviceUID(id),
              "in": streamLayout(id, input: true).reduce(0, +),
              "out": streamLayout(id, input: false).reduce(0, +),
              "rate": nominalSampleRate(id),
              "default_input": id == defIn, "default_output": id == defOut,
              "transport": transportType(id)])
    }
    exit(0)
}

let mic = MicCapture()
let sys = SystemCapture()
mic.preferredUID = inputUID
var micOK = false, sysOK = false
var synthTimer: DispatchSourceTimer?
if debugSynth {
    let q = DispatchQueue(label: "presage.synth", qos: .userInteractive)
    let t0 = DispatchTime.now().uptimeNanoseconds
    var made = 0
    var frame = [Int16](repeating: 0, count: FRAME_SAMPLES)
    let amp = 32768.0 * pow(10.0, -30.0 / 20.0)
    let timer = DispatchSource.makeTimerSource(queue: q)
    timer.schedule(deadline: .now(), repeating: .milliseconds(20), leeway: .milliseconds(1))
    timer.setEventHandler {
        let due = Int(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 * OUT_RATE)
            / FRAME_SAMPLES
        while made < due {
            for i in 0..<FRAME_SAMPLES {
                let n = Double(made * FRAME_SAMPLES + i)
                frame[i] = Int16(amp * sin(2 * Double.pi * 440 * n / OUT_RATE))
            }
            frame.withUnsafeBufferPointer { emitFrame(STREAM_MIC, $0.baseAddress!, FRAME_SAMPLES) }
            mic.stats.framesOut += FRAME_SAMPLES
            mic.stats.callbacks += 1
            mic.stats.peak = Int16(amp)
            made += 1
        }
    }
    timer.resume()
    synthTimer = timer
    micOK = true
    wantMic = false
    wantSystem = false
    emit(["event": "started", "stream": "mic", "device": "synth", "uid": "synth",
          "src_rate": OUT_RATE, "src_channels": 1, "transport": "virtual"])
} else {
    if wantMic { micOK = mic.start() }
    if wantSystem { sysOK = sys.start() }
}

if !micOK && !sysOK {
    emit(["event": "fatal", "reason": "no stream started"])
    exit(1)
}

// Neither permission fails loudly: a tap with System Audio Recording denied is created
// happily and returns silence, and a denied mic opens a stream whose callback never
// fires. Watch both and say so on stderr.
final class Watchdog {
    let name: String
    let stats: StreamStats
    var lastCallbacks = 0
    var lastFrames = 0
    var silentTicks = 0
    var state = "unknown"
    var everHadAudio = false
    init(_ name: String, _ stats: StreamStats) { self.name = name; self.stats = stats }

    /// One second of this stream, as the recovery logic needs to see it.
    @discardableResult
    func tick() -> (peak: Int16, moved: Bool) {
        let callbacks = stats.callbacks
        let frames = stats.framesOut
        let peak = stats.peak
        stats.peak = 0
        let movedCallbacks = callbacks != lastCallbacks
        let movedFrames = frames != lastFrames
        lastCallbacks = callbacks
        lastFrames = frames

        if peak > 0 {
            silentTicks = 0
            everHadAudio = true
            if state != "audio" { state = "audio"; emit(["event": "audio", "stream": name]) }
            return (peak, movedCallbacks || movedFrames)
        }
        silentTicks += 1
        guard silentTicks == 3, state != "silent" else { return (0, movedCallbacks || movedFrames) }
        state = "silent"
        let reason = movedCallbacks || movedFrames ? "zero_samples" : "no_callbacks"
        var o: [String: Any] = ["event": "silent", "stream": name, "reason": reason,
                                "callbacks": callbacks, "frames": frames]
        if name == "system" {
            o["hint"] = reason == "no_callbacks"
                ? "nothing is playing audio, so the tapped device is idle"
                : everHadAudio
                    // This tap has delivered real audio in this run, so the grant is
                    // there. The aggregate keeps running for a few seconds after the
                    // last process stops rendering, and those buffers are zeros.
                    ? "playback stopped: the tap keeps running briefly and delivers zeros"
                    : "tap is delivering silence: System Settings > Privacy & Security > Screen & System Audio Recording is probably not granted to the app that launched presage"
        } else {
            o["hint"] = reason == "no_callbacks"
                ? "input device is not delivering: no default input device, or the device is in use exclusively"
                : everHadAudio
                    ? "mic went digitally silent: Bluetooth headsets do this when the buds leave your ears or the link re-negotiates; recovering"
                    : "mic is delivering silence: muted, or System Settings > Privacy & Security > Microphone is not granted to the app that launched presage"
        }
        emit(o)
        return (0, movedCallbacks || movedFrames)
    }
}

var watchdogs: [Watchdog] = []
var micWatchdog: Watchdog?

/// The mic can start late: if the device it wanted was disappearing at launch (AirPods
/// do exactly that), the stream comes up on a later rebuild instead.
func ensureMicWatchdog() {
    guard micOK, micWatchdog == nil else { return }
    let w = Watchdog("mic", mic.stats)
    micWatchdog = w
    watchdogs.append(w)
}
ensureMicWatchdog()
if sysOK { watchdogs.append(Watchdog("system", sys.stats)) }

// A mic that delivers perfect digital zeros while Core Audio reports the device alive,
// running and unmuted is the AirPods failure mode (docs/research-audio.md, "mic auto
// recovery"): no device change, no rate change, nothing to react to, just silence for
// as long as the buds are out of your ears. Escalate instead of waiting it out.
//
//   3 s of zeros  ->  mic_state silent      (say so)
//                 ->  mic_state recovering  (stop/start the same device once)
//   3 s more      ->  mic_state fallback    (move to the built-in mic and stay)
//   audio again   ->  mic_state ok
//
// One cycle per 10 s at most; the fallback is held until the original device changes
// rate or the default input changes, either of which rebuilds the stream anyway.
final class MicRecovery {
    private var state = "unknown"      // unknown | ok | silent | recovering | fallback
    private var silentTicks = 0
    private var phase = 0              // 0 idle, 1 reopened once, 2 on the fallback
    private var lastCycleAt: Double = -1e9
    private var fromUID: String?
    private var fromName: String?
    private var fromDevice = AudioObjectID(kAudioObjectUnknown)
    private var fromRate: Double = 0

    private func emitState(_ s: String, _ extra: [String: Any] = [:]) {
        state = s
        var o: [String: Any] = ["event": "mic_state", "state": s, "device": mic.name, "uid": mic.uid]
        for (k, v) in extra { o[k] = v }
        emit(o)
    }

    /// Called once a second with the mic watchdog's peak for that second.
    func observe(peak: Int16, now: Double) {
        if peak > 0 {
            silentTicks = 0
            phase = 0
            if state != "ok" {
                var extra: [String: Any] = [:]
                if mic.isFallback {
                    extra["fallback"] = true
                    if let f = fromUID { extra["from"] = f }
                }
                emitState("ok", extra)
            }
            return
        }
        silentTicks += 1
        guard silentTicks >= 3 else { return }
        silentTicks = 0
        switch phase {
        case 0:
            guard now - lastCycleAt >= 10 else { return }
            lastCycleAt = now
            let dev = mic.devID
            let granted = micPermissionGranted()
            var detail: [String: Any] = ["running_somewhere": deviceIsRunningSomewhere(dev),
                                         "alive": deviceIsAlive(dev),
                                         "rate": mic.currentRate(),
                                         "muted": inputMuted(dev),
                                         "transport": transportType(dev)]
            if let v = inputVolume(dev) { detail["input_volume"] = v }
            if !granted {
                // Every device would be just as silent. Say so and stay put.
                detail["reason"] = "microphone_permission"
                detail["hint"] = "System Settings > Privacy & Security > Microphone is not granted to the app that launched presage"
                emitState("silent", detail)
                phase = 2
                return
            }
            emitState("silent", detail)
            let ok = mic.reopenSameDevice()
            micOK = micOK || ok
            emitState("recovering", ["reopened": ok])
            phase = 1
            if !ok { escalate(now: now) }
        case 1:
            escalate(now: now)
        default:
            // Already on the fallback and it is silent too. Nothing else to reach for;
            // the 10 s gate stops this from spinning.
            phase = 0
        }
    }

    /// Step 2: leave this device for the built-in mic and stay there.
    private func escalate(now: Double) {
        phase = 2
        lastCycleAt = now
        let leavingDevice = mic.devID
        let leavingUID = mic.uid
        let leavingName = mic.name
        guard let target = fallbackInputDevice(excluding: leavingDevice) else {
            emitState("silent", ["reason": "no_fallback_device"])
            return
        }
        fromUID = leavingUID
        fromName = leavingName
        fromDevice = leavingDevice
        fromRate = inputStreamFormat(leavingDevice)?.mSampleRate ?? nominalSampleRate(leavingDevice)
        if mic.switchTo(target, fallback: true) {
            micOK = true
            emitState("fallback", ["from": leavingUID, "from_device": leavingName])
        } else {
            micOK = false
            emit(["event": "mic_state", "state": "silent", "device": deviceName(target),
                  "uid": deviceUID(target), "reason": "fallback_failed", "from": leavingUID])
        }
    }

    /// While on the fallback, watch the device we left: a Bluetooth headset that
    /// re-negotiates its profile changes the input rate, and that is the signal it is
    /// worth going back. A device that vanishes changes the default input instead,
    /// which the hardware listener already rebuilds on.
    func checkOriginal() {
        guard mic.isFallback, fromDevice != AudioObjectID(kAudioObjectUnknown),
              inputChannelCount(fromDevice) > 0 else { return }
        let live = inputStreamFormat(fromDevice)?.mSampleRate ?? nominalSampleRate(fromDevice)
        guard live > 0, fromRate > 0, abs(live - fromRate) > 1 else { return }
        emit(["event": "rate_drift", "stream": "mic", "device": deviceName(fromDevice),
              "uid": deviceUID(fromDevice), "built": fromRate, "now": live,
              "note": "original mic changed rate, leaving the fallback"])
        fromRate = live
        reset()
        mic.clearPin()
        mic.stop()
        micOK = mic.start()
    }

    /// A real device change supersedes anything this state machine was doing.
    func reset() {
        state = "unknown"
        silentTicks = 0
        phase = 0
        fromDevice = AudioObjectID(kAudioObjectUnknown)
        fromUID = nil
        fromName = nil
    }
}

let micRecovery = MicRecovery()
var lastInputVolume: Double = -1
var lastMicMode = ""

let watchTimer = Timer(timeInterval: 1.0, repeats: true) { _ in
    let now = Date().timeIntervalSinceReferenceDate
    let c = frameOut.takeCounters()
    if c.mic > 0 || c.system > 0 {
        emit(["event": "overflow", "mic_frames": c.mic, "system_frames": c.system,
              "note": "the reader stopped for longer than the 30 s queue; the oldest audio was lost"])
    }
    // A call app's automatic mic level moves the shared device's input volume under us; a
    // mic that suddenly transcribes worse is often just this. Say so when it changes.
    if micOK, !debugSynth, mic.devID != AudioObjectID(kAudioObjectUnknown) {
        let v = inputVolume(mic.devID) ?? -1
        if abs(v - lastInputVolume) > 0.01 {
            if lastInputVolume >= 0 {
                emit(["event": "input_volume", "stream": "mic", "device": mic.name,
                      "from": lastInputVolume, "to": v])
            }
            lastInputVolume = v
        }
        let m = micMode()
        if m != lastMicMode {
            if !lastMicMode.isEmpty {
                emit(["event": "mic_mode", "stream": "mic", "from": lastMicMode, "to": m])
            }
            lastMicMode = m
        }
    }
    let streams = max(1, (micOK ? 1 : 0) + (sysOK ? 1 : 0))
    let behind = Double(c.peak) * Double(FRAME_SAMPLES) / OUT_RATE / Double(streams)
    if behind >= 0.5 {
        emit(["event": "backlog", "frames": c.peak, "seconds": (behind * 100).rounded() / 100,
              "note": "the reader fell behind; audio was queued, not lost"])
    }
    for w in watchdogs {
        let t = w.tick()
        if w === micWatchdog { micRecovery.observe(peak: t.peak, now: now) }
    }
    micRecovery.checkOriginal()
    checkRateDrift()
}
RunLoop.main.add(watchTimer, forMode: .common)

// --debug-mute-mic-after N: arm the fake silence for whatever device the mic is on now.
if debugMuteMicAfter > 0 {
    DispatchQueue.main.asyncAfter(deadline: .now() + debugMuteMicAfter) {
        mic.armDebugMute()
        emit(["event": "debug", "what": "mute_mic", "uid": mic.uid, "device": mic.name])
    }
}

/// Opening the mic is itself what makes a Bluetooth headset drop the input device from
/// 48 kHz to 24 kHz, so the change routinely lands between reading the rate and
/// registering the listener. Poll for it as well: a resampler built for the wrong rate
/// silently halves or doubles the audio without any error. Retune in place rather than
/// rebuilding the device, or this never settles.
func checkRateDrift() {
    if micOK, mic.builtRate > 0, !micRebuildPending {
        let now = mic.currentRate()
        if now > 0, abs(now - mic.builtRate) > 1 {
            emit(["event": "rate_drift", "stream": "mic", "built": mic.builtRate, "now": now])
            mic.retune(to: now)
        }
    }
    if sysOK, sys.builtRate > 0, !systemRebuildPending {
        let now = sys.currentRate()
        if now > 0, abs(now - sys.builtRate) > 1 {
            emit(["event": "rate_drift", "stream": "system", "built": sys.builtRate, "now": now])
            sys.retune(to: now)
        }
    }
}

// A different default device needs a full rebuild: new device, new IOProc. A device
// that merely changes rate does not — see retune(to:).
//
// One pending rebuild per stream. A burst of notifications during a Bluetooth switch
// collapses into a single rebuild; a repeating trigger cannot starve it, because a
// request that arrives while one is already pending is dropped rather than resetting
// the timer.
var micRebuildPending = false
var systemRebuildPending = false

func scheduleRebuild(_ why: String, mic rebuildMic: Bool, system rebuildSystem: Bool) {
    if rebuildMic {
        if micRebuildPending { return }
        micRebuildPending = true
    }
    if rebuildSystem {
        if systemRebuildPending { return }
        systemRebuildPending = true
    }
    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
        emit(["event": "device_changed", "why": why])
        if rebuildMic {
            micRebuildPending = false
            // A new default input supersedes a fallback: drop the pin and take the
            // device the user (or macOS) just chose, or --input-uid if one was given.
            micRecovery.reset()
            mic.clearPin()
            // Not `if micOK`: a stream that failed to start earlier — the device was
            // going away as we opened it — gets its chance here.
            if wantMic { mic.stop(); micOK = mic.start(); ensureMicWatchdog() }
        }
        if rebuildSystem {
            systemRebuildPending = false
            if wantSystem { sys.stop(); sysOK = sys.start() }
        }
    }
}

var rateListenedOutput = AudioObjectID(kAudioObjectUnknown)
var rateListenedInput = AudioObjectID(kAudioObjectUnknown)
var rateAddr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyNominalSampleRate,
                                          mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
// Registered whenever the stream was asked for, not only when it came up: a stream
// that failed to start is exactly the one that needs the next device change.
if wantSystem {
    var a = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                       mScope: kAudioObjectPropertyScopeGlobal,
                                       mElement: kAudioObjectPropertyElementMain)
    AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &a, DispatchQueue.main) { _, _ in
        scheduleRebuild("default_output_device", mic: false, system: true)
    }
    rateListenedOutput = defaultDeviceID(output: true)
    AudioObjectAddPropertyListenerBlock(rateListenedOutput, &rateAddr, DispatchQueue.main) { _, _ in
        checkRateDrift()
    }
}
if wantMic {
    var a = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice,
                                       mScope: kAudioObjectPropertyScopeGlobal,
                                       mElement: kAudioObjectPropertyElementMain)
    AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &a, DispatchQueue.main) { _, _ in
        scheduleRebuild("default_input_device", mic: true, system: false)
    }
    // Listen on the device we actually opened, which with --input-uid or after a
    // fallback is not the default input.
    rateListenedInput = mic.devID != AudioObjectID(kAudioObjectUnknown)
        ? mic.devID : defaultDeviceID(output: false)
    AudioObjectAddPropertyListenerBlock(rateListenedInput, &rateAddr, DispatchQueue.main) { _, _ in
        checkRateDrift()
    }
}

signal(SIGINT) { _ in exit(0) }
signal(SIGTERM) { _ in exit(0) }
signal(SIGPIPE, SIG_IGN)

if seconds > 0 {
    DispatchQueue.main.asyncAfter(deadline: .now() + seconds) {
        synthTimer?.cancel()
        if wantMic { mic.stop() }
        if wantSystem { sys.stop() }
        frameOut.drain(timeout: 1.0)
        emit(["event": "stopped"])
        exit(0)
    }
}
emit(["event": "ready", "mic": micOK, "system": sysOK, "rate": OUT_RATE, "frame_samples": FRAME_SAMPLES])
RunLoop.main.run()
