import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def compute_metrics(returns_series, risk_free_rate=0.02):
    """Computes Total Return, Annualized Return, Volatility, Sharpe Ratio, and Max Drawdown."""
    compounded = (1 + returns_series).cumprod()
    total_return = compounded.iloc[-1] - 1
    
    n_days = len(returns_series)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_vol = returns_series.std() * np.sqrt(252)
    
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / annualized_vol if annualized_vol != 0 else 0
    
    # Maximum Drawdown
    peak = compounded.cummax()
    drawdown = (compounded - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_vol * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100
    }

def main():
    tau = 0.025
    os.makedirs('figure', exist_ok=True)
    
    # Load all portfolio return series
    equal_weight = pd.read_csv('responses_portfolios/equal_weighted_portfolio.csv')
    opt_mvo = pd.read_csv('responses_portfolios/optimized_portfolio.csv')
    gemma_blm = pd.read_csv(f'results/gemma_black_litterman_returns_tau_{tau}.csv')
    llama_blm = pd.read_csv(f'results/llama_black_litterman_returns_tau_{tau}.csv')
    qwen_blm = pd.read_csv(f'results/qwen_black_litterman_returns_tau_{tau}.csv')
    
    df = pd.DataFrame({
        'Date': pd.to_datetime(equal_weight['Date']),
        'Equal Weighted (1/N)': equal_weight['Portfolio_Return'],
        'Mean-Variance (MVO)': opt_mvo['Portfolio_Return'],
        'Gemma-7B + BLM': gemma_blm['Portfolio_Return'],
        'LLaMA-3.1 + BLM': llama_blm['Portfolio_Return'],
        'Qwen2 + BLM': qwen_blm['Portfolio_Return']
    }).dropna()
    
    strategies = [
        'Equal Weighted (1/N)', 
        'Mean-Variance (MVO)', 
        'Gemma-7B + BLM', 
        'LLaMA-3.1 + BLM', 
        'Qwen2 + BLM'
    ]
    
    # 1. Compute Metrics Table
    metrics_list = []
    for strat in strategies:
        m = compute_metrics(df[strat])
        m['Strategy'] = strat
        metrics_list.append(m)
    
    metrics_df = pd.DataFrame(metrics_list).set_index('Strategy')
    print("\n" + "="*80)
    print("           OUT-OF-SAMPLE PERFORMANCE SUMMARY (July - Dec 2024)")
    print("="*80)
    print(metrics_df.round(2).to_string())
    print("="*80 + "\n")
    
    metrics_df.to_csv('results/performance_comparison_metrics.csv')
    
    # 2. Cumulative Returns Plot
    plt.figure(figsize=(12, 6))
    colors = {
        'Equal Weighted (1/N)': '#7f7f7f',
        'Mean-Variance (MVO)': '#ff7f0e',
        'Gemma-7B + BLM': '#2ca02c',
        'LLaMA-3.1 + BLM': '#1f77b4',
        'Qwen2 + BLM': '#9467bd'
    }
    
    for strat in strategies:
        cum_ret = (1 + df[strat]).cumprod() - 1
        plt.plot(df['Date'], cum_ret * 100, label=f"{strat} ({cum_ret.iloc[-1]*100:.1f}%)", 
                 linewidth=2.2, color=colors.get(strat))
        
    plt.title('Out-of-Sample Cumulative Returns Comparison (Jul 2024 - Dec 2024)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('figure/cumulative_returns_evaluation.png', dpi=300)
    plt.close()
    print("Saved: figure/cumulative_returns_evaluation.png")
    
    # 3. Bar Chart of Risk-Adjusted Metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_reset = metrics_df.reset_index()
    
    # Annualized Return
    sns.barplot(data=metrics_reset, x='Strategy', y='Annualized Return (%)', hue='Strategy', ax=axes[0], palette='Blues_d', legend=False)
    axes[0].set_title('Annualized Return (%)', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_ylabel('Return (%)')
    
    # Annualized Volatility
    sns.barplot(data=metrics_reset, x='Strategy', y='Annualized Volatility (%)', hue='Strategy', ax=axes[1], palette='Reds_d', legend=False)
    axes[1].set_title('Annualized Volatility (%)', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].set_ylabel('Volatility (%)')
    
    # Sharpe Ratio
    sns.barplot(data=metrics_reset, x='Strategy', y='Sharpe Ratio', hue='Strategy', ax=axes[2], palette='Greens_d', legend=False)
    axes[2].set_title('Sharpe Ratio (Risk-Adjusted)', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=30)
    axes[2].set_ylabel('Sharpe Ratio')
    
    plt.tight_layout()
    plt.savefig('figure/risk_adjusted_metrics.png', dpi=300)
    plt.close()
    print("Saved: figure/risk_adjusted_metrics.png")

if __name__ == '__main__':
    main()
