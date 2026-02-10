人民币兑欧元购汇策略 - Flask MVP
============================

功能
- 验证模式：用历史数据回测策略，展示交互式图表（可缩放/拖拽）+ 交易表（可滚动/搜索/排序）
- 实时模式：读取最新数据，基于固定计划周期给出“今日建议买入 EUR”并允许你手动记录实际成交
- 邮件提醒：当“策略建议买入”且“汇率达到你设置的阈值”时，发送邮件通知（在更新脚本中触发）

项目结构
- app.py：Flask 主入口
- strategy_model.py：你的策略算法（已移除示例主程序，便于 import）
- services/crawler.py：爬虫（由 spider.ipynb 整理为脚本）
- services/db.py：SQLite 存储
- services/strategy.py：回测 + 实时建议封装
- services/alerts.py：邮件提醒规则检查
- services/mailer.py：SMTP 发信
- templates/：页面模板
- static/：样式

快速启动
1) 安装依赖
   pip install -r requirements.txt

2) 启动
   python app.py

3) 打开浏览器
   http://127.0.0.1:5000

4) 首次使用先点“更新数据”，再进入验证/实时页面。

备注
- 数据源示例来自 kylc.com，实际部署请留意对方的使用条款与稳定性。
- 生产环境建议用 cron 定时跑更新，而不是依赖 Flask 进程（多进程下 APScheduler 容易重复跑）。

邮件提醒配置
-------------
1) 先在“实时模式”页面设置提醒邮箱与阈值汇率

2) 配置 SMTP 环境变量（示例）
   export SMTP_HOST="smtp.example.com"
   export SMTP_PORT="587"
   export SMTP_USER="your_account@example.com"
   export SMTP_PASS="your_app_password"
   # 可选
   export SMTP_FROM="your_account@example.com"
   export SMTP_USE_TLS="true"
   export SMTP_USE_SSL="false"

3) 运行更新脚本时会自动检查并发送提醒
   python update_rates.py

建议用 cron/任务计划每天跑一次 update_rates.py。
