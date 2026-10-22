## 背景と目的

<!-- 何が問題で、なぜ今直すのか。commit をまたぐ物語だけ書く（個々の Why は commit log にある） -->

## 変更内容

<!-- commit 単位の箇条書き。設計判断があれば理由を 1 行 -->

## 影響範囲

- 互換性: <!-- events.jsonl のイベント種別・フィールド / agent config のキーや CLI フラグ / 実験 JSON のスキーマ / checker の終了コードや analyze の JSON 出力 / state_dir（index.db, probe_progress, events.jsonl）の永続化形式、のどれかに影響するか。無ければ「なし」と根拠 -->
- 挙動が変わる利用者: <!-- 誰が何を見て気づくか -->

## 割り切り・見送り

<!-- scope 外にした指摘、既知の制約、後続の判断事項。CONTEXT.md / ADR へ書いたか -->

## 検証

- ローカル: <!-- unit（pytest 件数）/ ruff / mypy。ReCefore 上の実機ラン（root）を行った場合はその結果も -->
- 独立レビュー: <!-- 誰が（codex / pi）、指摘の扱い -->
- [ ] <!-- merge 後に確認すること（あれば） -->
