# ANSWER_KEY

Ground truth for the sample folder. The indexer skips any file whose name starts with
`ANSWER_KEY` or `.`, so this file is never retrievable and cannot leak into an answer.
All workbook figures are $ thousands unless the unit says otherwise.

## Acme_Financials.xlsx › P&L

Two title rows, a merged `FY25 actual` band on row 4 over a `Q1..Q4 / Total` header on
row 5, data from row 6. Column F (`Total`) and the gross margin / total opex / EBITDA
rows are formulas that carry cached values. Rows 9 and 15 (the two percent rows) are
formulas with **no cached value** and must come back NULL.

| Question | Answer | Citation |
| --- | --- | --- |
| Q3 FY25 revenue | 14,020 | Acme_Financials.xlsx › P&L › D6 |
| Q3 FY25 EBITDA | 1,773 | Acme_Financials.xlsx › P&L › D14 |
| FY25 total revenue | 54,880 | Acme_Financials.xlsx › P&L › F6 |
| FY25 EBITDA | 6,772 | Acme_Financials.xlsx › P&L › F14 |
| FY26 plan revenue | 68,500 | Acme_Financials.xlsx › P&L › G6 |
| FY26 plan EBITDA | 10,710 | Acme_Financials.xlsx › P&L › G14 |
| FY26 plan total opex | 34,500 | Acme_Financials.xlsx › P&L › G13 |
| FY25 total opex | 28,900 | Acme_Financials.xlsx › P&L › F13 |
| FY25 gross margin | 35,672 | Acme_Financials.xlsx › P&L › F8 |
| Gross margin % (any quarter) | NULL — formula with no cached value | Acme_Financials.xlsx › P&L › B9:G9 |
| EBITDA margin % (any quarter) | NULL — formula with no cached value | Acme_Financials.xlsx › P&L › B15:G15 |

Quarterly series (Q1, Q2, Q3, Q4):

- Revenue: 12,400, 13,150, 14,020, 15,310
- COGS: 4,340, 4,602, 4,907, 5,359
- Gross margin: 8,060, 8,548, 9,113, 9,951
- Total opex: 6,820, 7,080, 7,340, 7,660
- EBITDA: 1,240, 1,468, 1,773, 2,291

## Acme_Financials.xlsx › Headcount

Title on row 1, header on row 3, departments rows 4-8, `Total` row 9 (formulas, cached).

| Question | Answer | Citation |
| --- | --- | --- |
| Total headcount in January | 119 | Acme_Financials.xlsx › Headcount › B9 |
| Total headcount in June | 152 | Acme_Financials.xlsx › Headcount › G9 |
| Engineering headcount in June | 60 | Acme_Financials.xlsx › Headcount › G4 |
| Net adds Jan to Jun | 33 | derived from B9 and G9 |

Monthly totals (Jan, Feb, Mar, Apr, May, Jun): 119, 124, 131, 138, 144, 152

## Acme_Financials.xlsx › Regions

Two tables on one sheet. The first starts at C5 (columns A and B are empty), the second
at C12 after two blank rows.

| Question | Answer | Citation |
| --- | --- | --- |
| North America FY25 revenue | 30,184 | Acme_Financials.xlsx › Regions › D6 |
| EMEA FY25 revenue | 16,464 | Acme_Financials.xlsx › Regions › D7 |
| APAC FY25 revenue | 6,036 | Acme_Financials.xlsx › Regions › D8 |
| LATAM FY25 revenue | 2,196 | Acme_Financials.xlsx › Regions › D9 |
| North America customers | 96 | Acme_Financials.xlsx › Regions › F6 |
| EMEA customers | 58 | Acme_Financials.xlsx › Regions › F7 |
| APAC customers | 31 | Acme_Financials.xlsx › Regions › F8 |
| LATAM customers | 15 | Acme_Financials.xlsx › Regions › F9 |
| Enterprise FY25 revenue | 32,928 | Acme_Financials.xlsx › Regions › D13 |
| Mid-market FY25 revenue | 16,464 | Acme_Financials.xlsx › Regions › D14 |
| SMB FY25 revenue | 5,488 | Acme_Financials.xlsx › Regions › D15 |

Consistency checks that must hold:

- Regions FY25 revenue sums to 54,880 = P&L F6.
- Regions FY26 plan sums to 68,500 = P&L G6.
- Region customers sum to 200 = rows in customers.csv.
- Segment FY25 revenue sums to 54,880 = P&L F6.
- Segment logos sum to 200 = rows in customers.csv.

## customers.csv

200 rows. Columns: customer_id, customer_name, segment, region, arr_usd, churned,
renewal_month, csat. ARR is in dollars; segment ARR totals match the Regions sheet
segment revenue in thousands exactly.

| Question | Answer |
| --- | --- |
| Total ARR | $54,880,000 |
| Churned ARR (all segments) | $5,980,000 |
| Churned customer count | 16 of 200 |
| Enterprise: customers / ARR / churned / churned ARR / churn rate | 44 / $32,928,000 / 6 / $4,494,300 / 13.6% |
| Mid-market: customers / ARR / churned / churned ARR / churn rate | 71 / $16,464,000 / 5 / $1,168,200 / 7.0% |
| SMB: customers / ARR / churned / churned ARR / churn rate | 85 / $5,488,000 / 5 / $317,500 / 5.9% |

Enterprise churn (13.6%) is deliberately worse than mid-market (7.0%) and SMB (5.9%), so
"is churn getting worse in enterprise" has a real answer that agrees with Board_Memo.docx,
notes.md and the Strategy_Deck speaker notes.

## Documents

| File | What it says | Where |
| --- | --- | --- |
| Strategy_Deck.pptx | FY25 revenue, gross margin, EBITDA and the 12.3% margin | slide 2 |
| Strategy_Deck.pptx | Enterprise is 44 logos but 60% of ARR; NRR 118% vs 94% SMB | slide 3 |
| Strategy_Deck.pptx | Region table, every region planned at 25% growth | slide 4 |
| Strategy_Deck.pptx | Headcount 119 in January to 152 by June | slide 5 |
| Strategy_Deck.pptx | Speaker notes admit APAC cannot carry 25% growth | slide 4 notes |
| Board_Memo.docx | 12.3% actual vs 15% commitment; opex is the whole story | Summary |
| Board_Memo.docx | Quarterly EBITDA bridge table | Quarterly EBITDA bridge |
| Board_Memo.docx | 16 of 200 churned; 13.6% / 7.0% / 5.9% by segment | Churn and retention |
| Market_Report.pdf | TAM $14.2bn in 2025 to $21.6bn by 2028, 15% CAGR | page 1 |
| Market_Report.pdf | Four vendors hold 46% of the market | page 2 |
| Market_Report.pdf | Median NRR 104%, upper quartile 121%; gross retention 88% | page 3 |
| notes.md | The two percent rows came through blank | Open questions for the CFO |
| notes.md | Blended discount 29% vs market 22% | Pricing |
