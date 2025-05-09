# src/generate_portfolio_report.py

def generate_portfolio_notebook():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import Image, display
    from datetime import datetime
    import nbformat
    from nbformat import v4 as nbf

    # Paths (relative to notebooks/ directory)
    NOTEBOOK_FILE = "notebooks/portfolio_report.ipynb"
    SUMMARY_PATH = "../results/summary_backtest_2024.csv"
    IMAGES = {
        "winrate_vs_trades": "../results/debug_winrate_vs_trades.png",
        "avg_vs_total": "../results/debug_avg_vs_total_return.png",
        "return_dist": "../results/debug_return_distribution.png",
        "equity": "../results/debug_equity_curve.png",
        "monthly": "../results/debug_monthly_return_boxplot.png",
        "drawdown": "../results/debug_drawdown.png"
    }

    # Notebook cells
    nb = nbf.new_notebook()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    nb.cells = [
        nbf.new_markdown_cell("# 📊 Portfolio Report — Keltner Channel Strategy"),
        nbf.new_markdown_cell(f"*Report generated on: `{now}`*"),
        nbf.new_markdown_cell("This report provides a strategic overview of the Keltner-based trading strategy across a portfolio of stocks."),

        nbf.new_markdown_cell("## 🔹 Summary Table"),
        nbf.new_code_cell("import pandas as pd\nsummary = pd.read_csv('../results/summary_backtest_2024.csv')\nsummary.head()"),

        nbf.new_markdown_cell("## 🔹 Win Rate vs Number of Trades\nThis chart shows how frequently the strategy won (profitable trades) for each stock, plotted against the number of trades taken. Stocks in the upper-right corner have many trades and high win rates — ideal candidates."),
        nbf.new_code_cell("from IPython.display import Image\nImage('../results/debug_winrate_vs_trades.png')"),

        nbf.new_markdown_cell("## 🔹 Average Return vs Total Return\nThis plot shows how profitable each stock was, both on a per-trade basis (average) and in total. It highlights stocks with consistent and high-yielding trades."),
        nbf.new_code_cell("Image('../results/debug_avg_vs_total_return.png')"),

        nbf.new_markdown_cell("## 🔹 Distribution of Average Returns\nThis histogram illustrates how average returns are distributed across all stocks. It helps identify whether returns are skewed or tightly clustered."),
        nbf.new_code_cell("Image('../results/debug_return_distribution.png')"),

        nbf.new_markdown_cell("## 🔹 Simulated Equity Curve\nThis chart shows the simulated portfolio value over time if each trade had been executed sequentially. A rising curve indicates growing capital."),
        nbf.new_code_cell("Image('../results/debug_equity_curve.png')"),

        nbf.new_markdown_cell("## 🔹 Monthly Return Distribution\nEach box represents the spread of trade returns in a specific month. This shows whether the strategy performs better in specific months."),
        nbf.new_code_cell("Image('../results/debug_monthly_return_boxplot.png')"),

        nbf.new_markdown_cell("## 🔹 Drawdown Curve\nThis curve shows the worst-case decline from prior peaks. Shallow or infrequent drawdowns indicate more stable strategies."),
        nbf.new_code_cell("Image('../results/debug_drawdown.png')"),

        nbf.new_markdown_cell("## 🔹 Per-Ticker Trade Detail"),
        nbf.new_code_cell(
            "# Replace 'AAPL' with any ticker to view that stock's trade log\n"
            "ticker = 'AAPL'\n"
            "pd.read_csv(f'../results/{ticker}_trades.csv').head()"
        )
    ]

    # Save the notebook
    with open(NOTEBOOK_FILE, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Updated portfolio report notebook created: {NOTEBOOK_FILE}")
