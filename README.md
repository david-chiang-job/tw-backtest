# tw-backtest

## 用途

回測還原權息的台股加權報酬指數的框架。
你用 TOML 設定檔寫「什麼條件下買進、什麼條件下賣出、買賣到多少股票比例」，框架在整段歷史的每個交易日開盤各自空倉起跑，算出一批窗口的績效並排名。

不做 ETF、個股、除息與分割帳；只處理報酬指數，股價指數（不含息）只用來換算盤中的開高低；不附行情、不抓資料。

主要能力：

- 兩種內建指標：`ma_deviation`（均線乖離）與 `drawdown`（距前高）。
- 三種條件組合 `any`、`all`、`weighted`，可以巢狀。
- 兩種成交：觸價成交 `touch`，以及收盤觸發、隔日開盤成交 `close_next_open`。
- 三種重置規則：`recross`、`new_high`、`none`。
- 固定期窗口與多空循環窗口。
- 參數範圍 `{ from, to, step }` 自動展開成多組策略，用 numba 平行掃描。

## 安裝

需要 Python 3.11 以上。指令用 Windows 的 `py`。

當套件裝進自己的專案：

```text
py -m pip install "git+https://github.com/david-chiang-job/tw-backtest@v0.1.0"
```

要跑範例：clone 固定版本標籤再裝。

```text
git clone --branch v0.1.0 https://github.com/david-chiang-job/tw-backtest
cd tw-backtest
py -m venv .venv
.venv/Scripts/python -m pip install .
```

安裝後有 `twbacktest` 命令（在 venv 裡是 `.venv/Scripts/twbacktest`）。

## 範例（快速開始）

接著上面的 clone 與安裝，在 clone 出來的資料夾裡：

```text
.venv/Scripts/python -m twbacktest.synthetic --out examples/data
.venv/Scripts/twbacktest run examples/basic.toml --out examples/out
```

第一行產生合成行情到範例設定檔讀的 `examples/data/`；第二行跑回測，印出組數、耗時與前幾名策略，並在 `examples/out/` 寫出 `scores.csv` 與 `windows.csv`（欄位見〈輸出〉）。
`examples/data/` 與 `examples/out/` 不進 git。

範例設定檔 `examples/basic.toml` 示範：一個買進檔位（`weighted` 加權，其中一個門檻寫成範圍，展開成 3 組策略）、一個賣出檔位（`any`），以及 `[account]`、`[costs]`、`[fill]`、`[run]`、`[windows]` 各段的寫法，註解說明每段在做什麼。
範例的參數只是示意，排名也只是合成行情上的結果，不代表任何結論。複製一份改參數就能用。

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

## 設定檔

設定檔是 TOML。除了 `[[tiers]]` 必填，其他段落都可省略，省略就用下列預設值。
`[account]`、`[costs]`、`[run]`、`[fill]`、`[windows]` 與檔位內不認得的鍵會直接報錯，錯誤訊息會寫出位置與可用的選項。

### `[data]`

| 鍵 | 型別 | 預設 | 說明 |
| --- | --- | --- | --- |
| `dir` | 字串 | `"."` | 行情資料夾，相對於設定檔所在的資料夾 |
| `price_index` | 字串 | `"taiex.csv"` | 股價指數檔名 |
| `total_return` | 字串 | `"taiex-total-return.csv"` | 報酬指數檔名 |

預設檔名就是合成行情的檔名。檔案格式見〈資料格式〉。

### `[account]`

| 鍵 | 型別 | 預設 | 說明 |
| --- | --- | --- | --- |
| `cash` | 數字 | `5000000` | 起始本金，必須大於 0 |
| `cash_rate` | 數字 | `0.015` | 現金年利率，必須大於 -1 |

空倉起跑。現金在相鄰交易日之間按 `(1 + 年利率) ^ (相隔曆日 / 365)` 計息。

### `[costs]`

| 鍵 | 型別 | 預設 | 說明 |
| --- | --- | --- | --- |
| `buy_fee` | 數字 | `0.00049875` | 買進手續費率 |
| `sell_fee` | 數字 | `0.00049875` | 賣出手續費率 |
| `sell_tax` | 數字 | `0.001` | 賣出證交稅率 |

三者都必須在 0 到 1 之間（不含 1）。費稅按成交金額比例收，不計滑價。
台股牌價是手續費 0.001425、證交稅 0.001。

### `[run]`

| 鍵 | 型別 | 預設 | 說明 |
| --- | --- | --- | --- |
| `threads` | `"auto"` 或正整數 | `"auto"` | 平行執行緒數 |
| `keep_windows` | 0 以上整數 | `10` | 只為分數前 N 名另存逐窗結果（見〈輸出〉） |

### `[fill]`

| 鍵 | 型別 | 預設 | 說明 |
| --- | --- | --- | --- |
| `model` | 字串 | `"touch"` | 成交模型，`touch` 或 `close_next_open` |
| `same_day` | 字串 | `"buy_first"` | 同一天買賣都觸發時，`buy_first` 先買或 `sell_first` 先賣 |
| `band` | `[下, 上]` | `[0.10, 0.90]` | 只有 `touch` 用，需滿足 0 ≤ 下 ≤ 上 ≤ 1 |

觸價成交 `touch`：以當天最低價為 0、最高價為 1，只有 `band` 區間內的價位，加上開盤與收盤碰到的價位算數，影線最外側那段不算。
設觸發價為 P、振幅為 高 − 低：

- 下緣 = min(開, 收, 低 + 下 × 振幅)，上緣 = max(開, 收, 低 + 上 × 振幅)。
- 買進在下緣 ≤ P 時觸發；開盤 ≤ P 以開盤價成交，否則以 P 成交。
- 賣出在上緣 ≥ P 時觸發；開盤 ≥ P 以開盤價成交，否則以 P 成交。
- 開盤已經越過 P 是跳空，掛單在開盤成交。

例：某天低 100、高 120、開 105、收 110，`band = [0.10, 0.90]`，下緣 = min(105, 110, 102) = 102。買進觸發價 103 會觸發，以 103 成交；觸發價 101 只有影線最外側碰到，不觸發。

`close_next_open`：前一天收盤越過觸發價（買進 ≤、賣出 ≥），當天開盤成交。

同一天多檔觸發：買進依觸發價由高到低成交、賣出由低到高，後一檔用前一檔成交後的現金與持股；同一天買賣都觸發則依 `same_day`。

### `[[tiers]]`：檔位

檔位 = 條件 + 目標股票比例 + 重置規則。至少要有一個，每個檔位是一個 `[[tiers]]`。

| 鍵 | 型別 | 說明 |
| --- | --- | --- |
| `name` | 字串 | 檔位名稱，自訂 |
| `target_stock` | 0 到 1 的數字 | 目標股票比例 |
| `reset` | 字串 | 重置規則：`recross`、`new_high`、`none` |
| `when` | 表 | 觸發條件，寫成 `[tiers.when]` |

買進檔位觸發時把股票比例補到目標，已達標不動；賣出檔位只賣到目標。
同一檔位的條件只能同一個方向：`at_most` 是買進、`at_least` 是賣出，不能混用。

重置規則：

- 起跑日每個檔位都待命；有成交才進入等待。碰到觸發價但已達目標沒下單，就維持待命。
- `recross`：收盤回到當天觸發價的另一側（買進收盤高於觸發價、賣出收盤低於觸發價），隔天起待命。
- `new_high`：收盤不低於前一個交易日為止的最高收盤，隔天起待命。
- `none`：成交隔天就待命。

#### `[tiers.when]`：條件

條件是單一指標條件，或一種組合（可巢狀）。

單一指標條件的鍵：

| 鍵 | 說明 |
| --- | --- |
| `indicator` | 指標名稱 |
| 指標參數 | 例如 `ma_deviation` 的 `days` |
| `at_most` 或 `at_least` | 門檻，恰好寫一個 |
| `points` | 只能用在 `weighted` 的子條件 |

指標都是價位型：每天開盤前用前一天為止的報酬指數收盤，算出當天的觸發價。

- `ma_deviation`（參數 `days`）：前 `days` 個交易日收盤的平均 ×（1 + 門檻）。
- `drawdown`（無參數）：前一個交易日為止的最高收盤 ×（1 + 門檻）。
- `at_most` 是價格跌到觸發價以下才成立（買進用），`at_least` 是漲到觸發價以上才成立（賣出用）。
- 算不出觸發價的日子（例如行情還不到 `days` 天）是 NaN，該條件當天不成立。

組合（以下以買進方向說明，賣出方向高低對調）：

- `any = [...]`：各觸發價中最高的；NaN 略過，全部 NaN 當天不觸發。
- `all = [...]`：各觸發價中最低的；有任一 NaN 當天不觸發。
- `weighted = [...]` 加 `score_at_least = N`：子條件各有 `points`，觸發價由高到低累加分數，第一次累積到 N 的那個價位就是觸發價；NaN 不給分，湊不到 N 當天不觸發。

載入設定時會擋下：同一檔位買賣條件混用、`points` 或 `score_at_least` 不大於 0、全部分數加起來到不了 `score_at_least`。

例：三個子條件分數 3、2、2，`score_at_least = 4`。觸發價（買進）由高到低是 90、85、80，累積分數 3、5、7，第一次到 4 是在 85，觸發價就是 85。

巢狀的寫法：

```toml
[tiers.when]
all = [
  { indicator = "drawdown", at_most = -0.30 },
  { any = [
    { indicator = "ma_deviation", days = 100, at_most = -0.05 },
    { indicator = "ma_deviation", days = 30, at_most = -0.10 },
  ] },
]
```

### `[windows]`

每個交易日開盤各自空倉起跑，不同窗口在不同的終點結算。不寫這段就用預設：

```toml
[windows]
fixed = { lengths = [240, 720, 1200] }
cycles = { counts = [1, 2, 3], drawdown = 0.20 }
```

- `fixed`（參數 `lengths`）：固定期，N 個交易日窗口在第 N 個交易日收盤結算；行情不夠長的起跑日不產生。
- `cycles`（參數 `counts`、`drawdown`）：多空循環。報酬指數收盤從前高跌至少 `drawdown`、再收復前高算一場；`counts` 的 1 至 3 在第 1 至 3 場收復日收盤結算。
- 窗口產生器的參數也可以直接寫清單，例如 `fixed = [240, 720]` 等於 `fixed = { lengths = [240, 720] }`。
- `score_end`（`YYYY-MM-DD` 字串）：結算日晚於這天的窗口不進評分。

只要寫了 `[windows]` 就以你寫的產生器為準，沒寫到的產生器不會跑；只寫 `score_end` 則用預設的兩種窗口。

另外有「開場全倉基準」：同本金、同費稅，起跑日開盤買到可成交的最大單位，所有策略共用，在 `windows.csv` 以 `baseline` 列出，用來算超額報酬。

### 參數範圍

任何數字都可以寫成範圍 `{ from = 起, to = 迄, step = 間隔 }`，三個鍵缺一不可。

- 包含起迄兩端；`step` 的方向要跟 `to - from` 一致、不可為 0。
- 三個值都是整數時結果是整數；有小數時每個值四捨五入到小數 10 位。
- 設定檔裡有多個範圍時取笛卡兒積：例如 3 個值乘 4 個值是 12 組策略。每組各自檢查一次，錯誤訊息會寫出是第幾組、各範圍取了什麼值。
- 範圍可以寫在 `[[tiers]]` 與 `[fill]` 裡。
- `[account]`、`[costs]`、`[windows]`、`[run]` 整批策略共用，不同就會報錯，所以不能寫成範圍。

## 輸出

`twbacktest run <設定檔> --out <資料夾>` 在資料夾寫出兩個 CSV（UTF-8），並在終端機印出組數、每塊組數、執行緒數、耗時，以及前 5 名。

`scores.csv`：每組策略一列，依主排名分數由高到低（NaN 排最後）。

- `rank`：名次；`strategy`：策略編號（從 0 起，依展開順序）。
- 接著每個有寫成範圍的位置一欄，欄名是設定檔內的路徑（例如 `tiers.0.when.weighted.1.at_most`），值是這組取的值。
- 最後是評分欄，欄名是 `窗口種類.指標.評分方式`，預設共 13 欄：
  `720d.excess_total.median`（主排名）、`720d.total_return.p10`、`1200d.excess_total.median`、`1200d.total_return.p10`、`240d.excess_annual.median`、`240d.wait_days.nan_share`、`240d.wait_days.median`，以及 `cycle1`、`cycle2`、`cycle3` 各自的 `total_return.group_mean` 與 `excess_total.group_mean`。
  該種類沒有窗口時是 NaN。

評分方式：`median` 是忽略 NaN 的中位數、`p10` 是第 10 百分位、`nan_share` 是 NaN 的比例（套在 `wait_days` 上就是從沒買入的窗口比例）、`group_mean` 是先在同一場循環內平均、再把各場等權平均。

逐窗指標：

- `total_return`：期末淨值 ÷ 本金 − 1。
- `annualized`：年化報酬，以 365.2425 天為一年。
- `excess_total`、`excess_annual`：跟同窗全倉基準的差，單位是百分點。
- `max_drawdown`：最大回落。
- `principal_loss`：本金最大虧損比例，沒虧到本金是 0。
- `wait_days`：起跑日到首次買入的曆日數，沒買入是 NaN。

`windows.csv`：只含分數前 `keep_windows` 名的策略（`strategy` 欄是策略編號）與 `baseline`，每個窗口一列。
欄位：`strategy`、`kind`（`240d`、`cycle1` 等）、`start`、`end`、`cycle`（循環窗口的分組日期，固定期為空），後面接上述七個逐窗指標。

重新評分：

```text
.venv/Scripts/twbacktest score examples/out/windows.csv [--by 分數欄] [--out 檔案]
```

只用 `windows.csv` 重算評分與排名，不用行情與設定檔；不加 `--out` 就印到終端機，`--by` 預設是主排名欄。
`baseline` 會算分數但不進名次。

## 合成行情

用固定種子產生合成行情，供範例與測試使用，同一種子輸出逐位元組相同：

```text
py -m twbacktest.synthetic --out <資料夾> [--days N] [--seed S]
```

會在資料夾寫出 `taiex.csv` 與 `taiex-total-return.csv`。

## 怎麼加一種新指標

指標是登記表裡的名字，設定檔用名稱選。
在 `src/twbacktest/indicators.py` 用 `@register_indicator("名稱")` 登記一個基準函式：

- 第一個參數是整段報酬指數收盤（numpy 陣列），其餘參數是指標參數（不含門檻），會從設定檔的同一個條件表讀進來。
- 回傳跟收盤一樣長的陣列：第 t 天的值只能用第 t 天以前的收盤，算不出的日子填 `np.nan`。
- 觸發價預設 = 基準 ×（1 + 門檻）；要別的套法，傳 `apply_threshold=函式(基準, 門檻)`。

最小例子：前一天收盤（門檻 -0.05 就是比前一天收盤低 5% 的價位）。

```python
@register_indicator("prev_close")
def prev_close(close):
    out = np.full(len(close), np.nan)
    out[1:] = close[:-1]
    return out
```

設定檔就能用：

```toml
[tiers.when]
indicator = "prev_close"
at_most = -0.05
```

別忘了在 `tests/` 補測試。
條件組合、成交模型、重置規則、窗口產生器、評分也是同一套登記表（`registry.py`），各自在對應模組用 decorator 登記名稱。

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
