# Dataset information

The study uses third-party daily market data. Raw third-party files are not redistributed in this public-ready package because access and redistribution are governed by the respective providers.

## Required local filenames

Place the four prepared CSV files in this `data/` directory using these names:

- `CSI300_daily_data.csv`
- `SP500_daily_data.csv`
- `CSI1000_daily_data.csv`
- `BTC_USD_daily_data.csv`

The experiment runners require `Date` and `Log_Return` columns. The optional return-verification script also requires the corresponding `Close` column so it can recompute log returns. After placing the files, run `python code/check_data.py` and `python code/verify_returns.py` to verify the local inputs.

## Original third-party sources reported in the manuscript

- CSI 300 and CSI 1000: Wind Financial Terminal  
  https://www.wind.com.cn/portal/zh/WFT/index.html
- S&P 500 historical data, ticker `^GSPC`: Yahoo Finance  
  https://finance.yahoo.com/quote/%5EGSPC/history/
- BTC/USD: Coin Metrics reference-rate documentation  
  https://docs.coinmetrics.io/market-data/reference-rates-overview/reference_rate

Please follow each provider's licensing and access terms. The manuscript states the exact sample periods and price-field conventions used in the reported analysis.
