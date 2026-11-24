# tw-backtest

## 用途

回測還原權息的台股加權報酬指數的框架。
只處理報酬指數；股價指數（不含息）只用來換算盤中的開高低。
不做 ETF、個股、除息與分割帳。

## 資料格式

框架不附任何行情，也不附抓資料的腳本，請自備兩份 CSV（UTF-8，容許 BOM，日期為 `YYYY-MM-DD`，由舊到新）：

- 股價指數：`date,open,high,low,close`
- 報酬指數：`date,close`

載入時會檢查，任一項不符就丟 `DataError`，訊息寫出檔名、日期或行號與原因：

- 標頭必須恰好是上面的欄位。
- 日期在各檔內嚴格遞增，不可倒序或重複。
- 所有數值都要有，且是有限正數。
- 報酬指數的每個交易日，股價指數同一天都要有，且最低 ≤ 開、收 ≤ 最高。
- 股價指數在報酬指數第一天以前的列忽略（日期與數值格式仍要正確，不檢查開高低關係）；從報酬指數第一天起，兩檔的日期必須完全一致。

回測從報酬指數第一天起算。
報酬指數的開高低用當天的比例換算：`報酬指數最高 = 股價指數最高 × 報酬指數收盤 ÷ 股價指數收盤`，收盤用原值。
載入結果會記下每個檔案的檔名、筆數、起訖日期與 SHA-256。

```python
from twbacktest.data import load_market

market = load_market("股價指數.csv", "報酬指數.csv")
```

## 合成行情

用固定種子產生合成行情，供範例與測試使用，同一種子輸出逐位元組相同：

```text
py -m twbacktest.synthetic --out <資料夾> [--days N] [--seed S]
```

會在資料夾寫出 `taiex.csv` 與 `taiex-total-return.csv`。

## 開發

需要 Python 3.11 以上。

```text
py -m venv .venv
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m unittest
```

只跑某一支：`.venv/Scripts/python -m unittest tests.test_data`。

## 授權

MIT，見 `LICENSE`。
