# Semantic Index 初期ベースライン

## Destination

断続的なマルチホップ ICN（Cefore v0.12.0 / CCNx 1.0）で、各ノードが Semantic Index を持ち、Index record を隣接ノードとの pull 同期で全複製する。Index origin が一時的に到達不能でも Replica holder 経由で記録が伝わり、欠落・順序逆転・重複・停止・再起動・切断と復旧から回復できることを、ReCefore 上の実 Interest/Content Object で示す。受け入れ試験は T1 と T2（下記）。性能優位や新規性の証明は目的に含めない。

## Notes

- 実行（execution）型の effort。計画だけでなく実装と検証まで進める。
- 用語は `CONTEXT.md`、構造的な決定は `docs/adr/`（0001: リポジトリ分離、0002: holder スコープ pull 同期）。
- 名前の例はすべて暫定のワイヤ形式。確定はチケット 05 で行い、この map に追記する。
- 対象外（初期ベースラインから切り離すもの）: Cefore のヘッダ追加と転送改造、プラグイン、選択的複製、キャッシュ置換の最適化、Jev/VLM の学習と Jetson 推論、自然言語検索、未知ノードの自動発見、本番用の信頼管理と署名、恒久分断中の全体同期、経路も複製も無いコンテンツの取得保証、削除と有効期限の伝搬、遠隔 lookup。

## Decisions so far

### 環境と構成
- 試験は mininet-debian 上の Mininet + ReCefore。hpc-debian（RTX PRO 5000）はモデル訓練専用で、この effort では使わない。
- リポジトリは `/home/lab_shared/cefore-semindex`（remote `git@github.com:ss-lab-tut/cefore-semindex.git`）。最初の骨組みだけ org main に直接 push し、以後は個人フォークのブランチから PR。
- 言語は Python 3.12、uv 管理。Cefore 束縛は cefpyco（上流最新 v0.10.1）を `/home/lab_shared/cefpyco` に clone し、v0.12.0 で失敗した箇所だけ個人フォークのブランチで直す。cmake は apt ではなく venv 内の PyPI wheel で賄う（システム変更を避ける）。
- cefpyco の既知欠陥は使い方で避ける: `receive()` が GIL を握ったまま待つ・受信バッファが process-wide static なので、エージェントは単一スレッドのイベントループ（receive の timeout は 100ms 程度）。`send_data` はペイロード長を検査しないので、こちらの層で上限を検査してから渡す。
- `create_handle(ceforedir=<cefnetd.conf があるディレクトリそのもの>, portnum=0)` でノードごとの cefnetd を選ぶ。`register(prefix)` はプレフィックス一致（`cef_client.c:616-617`）。

### ReCefore 側に足す汎用機能（同期の知識は入れない、ADR-0001）
- `host_apps:` ホストごとの長寿命プロセス。argv/env/cwd にプレースホルダ（ノード名、ノードディレクトリ、cefnetd.conf のディレクトリ、シナリオの t0 epoch ms）を展開する。cefnetd 起動後に `CommandRunner.start` で起動し、`shutdown_runtime_resources` の中で cefnetd を止める前に終了させる。
- `links:` disaster シナリオでトポロジを明示する（無ければ従来の乱数 mesh）。
- `prefix_routes: {prefix: host}` 既存の FIB 生成（`compute_fib_for_uris`）に足し込む。put/pubsub_pub があると既定プレフィックスが消える既存の性質とは独立に効く。

### 時刻の基準
- ReCefore は scheduler の `at: 0` に当たる epoch ms を `host_apps` に渡す。エージェントの予定（レコード生成、probe）はすべてこの t0 からの相対秒で書く。
- 1つの実験記述から、ReCefore の YAML と各ノードのエージェント設定を生成スクリプトが同時に出力する。put の URI とレコードの content name は、この生成器が同じ値から作るので食い違わない。

### 同期プロトコル（ADR-0002）
- pull 型、同期相手は設定した直接隣接ノードのみ。Interest ペイロードとヘッダ便乗は使わない。
- 名前（暫定）: 状態確認 `ccnx:/semidx/<group>/<holder>/sv/<requester>/<epoch_ms>`、レコード `ccnx:/semidx/<group>/<holder>/rec/<origin>/<seq>`。各エージェントは `ccnx:/semidx/<group>/<自ノード>` を1つだけ register する（app_reg は先勝ちの線形走査なので、重なるプレフィックスを複数登録しない）。
- 同期用の Interest/Data には chunk TLV を付けない。コンテンツ本体の取得だけ chunk 0..end を使う（cefputfile が作ったオブジェクト）。
- Holder state: origin ごとに `hi`（1 から連続保持の最大 seq）と `extra`（hi より上の飛び飛びの保持 seq、昇順、最大 64）。応答は正規形 JSON で 8 KiB 以下、超えたら extra を切り詰めて `truncated: true`、hi は必ず送る。ページングは無し。
- 要求側は「通知された最大 seq」（隣接ノードの hi と extra の最大）、「連続して保持している範囲」、「未取得」を区別して持つ。1、2、4 を受け取っても 3 を取得済みにしない。

### Index record と永続化
- 項目: origin、seq、content_name、attrs（zone `zone1`〜`zone5`、disaster `flood`/`fire`/`landslide`/`collapse`/`unknown`、media `image`/`video`/`sensor`、いずれも単一値）、observed_at、schema_version=1、model_id=`sample-v0`、supersedes（スキーマにだけ用意、初期試験では使わない）。語彙はスキーマ定義側に置く。
- 符号化: キー整列・空白なしの UTF-8 JSON を Data ペイロードに入れる。1 件 2 KiB 以下。content_hash は受信側がこのバイト列の SHA-256 として計算する（レコードには入れない）。名前の origin/seq とペイロードの値が違えば拒否。
- 検証はスキーマ検証と衝突検出まで。署名と鍵配布は実装せず、文書に「未認証」と明記する。
- 永続化は SQLite（WAL）。レコード挿入と進捗更新は同じトランザクション。追記専用。
- origin ID は設定で静的に与えるノード名。seq は SQLite で採番し、再起動後は保存済み最大 + 1 から。同じ (origin, seq) で中身が違うレコードは先着を残し `conflict` として記録する。

### 時間・サイズの初期値（設定で変更可）
| 項目 | 値 |
|---|---|
| 状態確認の周期 | 2 秒 ±20% |
| Interest lifetime | 2000 ms |
| レコード取得の再送間隔 | 1 秒から倍々、16 秒で頭打ち |
| 隣接ノード 1 つあたりの同時取得数 | 8 |
| origin ごとの未取得管理上限 | 1024 |
| 状態 Data | expiry 1 秒、cache_time 0 |
| レコード Data | expiry 3600 秒、cache_time 300 秒 |

### 実装の構造
- 同期の中身（レコード、ストア、Holder state、未取得の計算、再送とバックオフ）は transport に依存しない純粋な部品にし、in-memory の fake transport（分断も表現できる複数ノード網）で root なしの単体試験を回す。cefpyco の adapter が唯一の実 transport。
- 試験用の故障注入: holder 側の設定 `fault.drop_rec` で、指定した (origin, seq) の応答を指定回数だけ返さない。本番の設定には現れない。

### 評価の記録
- 各ノードのエージェントが `hN/semindex/events.jsonl` に 1 行 1 イベントで書く: `record_created`、`state_rx`、`record_stored`（origin、seq、holder）、`fetch_timeout`、`conflict`、`probe_discovered`、`probe_retrieved`。時刻は壁時計（Mininet のホストは時計を共有）。送受信バイト数もエージェント側で数える。
- 解析スクリプト（このリポジトリ）が発見遅延（origin の record_created → 各ノードの record_stored）、全ノード一致時刻、制御通信量を出す。ReCefore の results.json には何も足さない。

### 受け入れ試験
- **T1（中継複製）**: 直線 h0–h1–h2（A–B–C）。h0 の本体を ReCefore の put で公開し、h0 のエージェントが N 件のレコードを生成する。全件が h2 に届いたあと h0–h1 を link_down し、h0 が追加で M 件を生成する。
  - 切断中の probe（h2）: 切断前に伝わった N 件の発見が成功すること。本体取得の結果は記録するが合否に使わない（Publisher の h0 に届かないため）。
  - link_up 後: 全ノードの索引が N+M 件で一致すること。h2 の probe で発見と本体 1 件の取得が成功すること。発見と取得は別々に記録する。
- **T2（欠落回復と再起動）**: `fault.drop_rec` で seq 3 の応答を落とし、再送で連続範囲が埋まること。順序逆転と重複を処理すること。エージェントを停止・再起動しても seq が衝突せず、進捗が先行しないこと。

### 進め方
- worker は sonnet5.5 xhigh（機械的な作業は haiku5.5）。各段階の監査は worker と別系統（codex CLI read-only または pi）。段階ごとの総括レビューは fable5.1。worker がさらに subagent を呼ぶことは禁止。
- sudo を要する Mininet の実行はオーケストレータ（メインスレッド）が行い、worker はコードと単体試験を渡す。
- 進捗は `PLAN.md` とボード（非公開 Artifact）。5 時間の利用上限に備えて wakeup を用意する。

## Fog

- cefnetd の CS 分岐（`cef_netd.c:3673-3740`、CS_MODE=2 で csmgrd がいるノード）で、プレフィックス登録したアプリに Interest が届くか。ソースを読む限り、完全一致を要求するのは `from_pub_f` 付きの Interest だけに見えるが、実機では未確認。チケット 01 の疎通試験で、csmgrd があるノードと無いノードの両方で確かめる。ここが否定されたら、転送の設計（名前の付け方、登録の仕方）を見直す。
- cefpyco v0.10.1 が Cefore v0.12.0 でそのまま動くか（チケット 01）。
- ReCefore のテンプレートは h0〜h2 に別々の `LOCAL_SOCK_ID` を持つ。h3 以降でも衝突しないか（5 ノード評価のとき）。
- 5 ノード評価の分母、反復数、切断率は T1/T2 の合格後に決める。
