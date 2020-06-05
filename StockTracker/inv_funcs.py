import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import datetime as dt
import yfinance as yf
from matplotlib import style
from dateutil.relativedelta import relativedelta


def remove_symbol_change_float(df):
    """
    Input a DataFrame
    For the columns: Cost Basis, Price, and Market Value,
    Remove the '$' and ',' symbol
    Change the data type to 'float'
    """
    dollar_list = ['Cost Basis', 'Price', 'Market Value']

    for i in dollar_list:
        df[i] = df[i].str.replace('$', '').str.replace(
            ',', '').str.strip().astype(float)

    return df


def update_price_values(df):
    """
    Input a DataFrame
    Update Tickers' Price and Market Values
    """
    start = (dt.datetime.today() - dt.timedelta(days=7)).strftime("%Y-%m-%d")
    today = dt.datetime.today().strftime("%Y-%m-%d")
    close_price = []
    tickers = df['Symbol'].tolist()

    data = yf.download(tickers=tickers, start=start, end=today)
    close_symbols = data.iloc[-1]['Close']
    new_price = pd.DataFrame({'Symbol': close_symbols.index.tolist(),
                              'Close_Price': close_symbols.values
                              })

    new_price['Close_Price'] = new_price['Close_Price'].round(2)

    df = df.sort_values(by='Symbol')
    new_price = new_price.sort_values(by='Symbol')
    df = df.reset_index()
    df = df.drop('index', axis=1)
    df.update(new_price)
    df['Market Value'] = df['Quantity'] * df['Close_Price']
    df['Prev_Price'] = data.iloc[-2]['Close'].values.round(2)
    df['Price_Chg'] = df['Close_Price'] - df['Prev_Price']
    df['Pct_Chg'] = (df['Price_Chg'] / df['Close_Price'] * 100).round(2)
    return df


def portfolio_comparison(df):
    """
    Input a DataFrame
    Return a Bar plot of portfolio distribution in comparison to Schwab BMI and Buffett
    """
    sectors = ["Communication", "Consumer Discretionary", "Consumer Staples",
               "Energy", "Financials", "Health Care", "Industrials",
               "IT", "Materials", "Real Estate", "Utilities"]
    bmi_percent = [0.081, 0.1090, 0.0760, 0.0450, 0.1620,
                   0.1150, 0.1150, 0.1720, 0.0490, 0.0420, 0.034]
    buffett_pct = [0.027, 0.0191, 0.1426, 0.0053, 0.4332,
                   0.0148, 0.0421, 0.3100, 0.0030, 0.0029, 0.000]

    # DataFrame for Schwab BMI Portfolio Distribution
    bmi_df = pd.DataFrame({"Sector": sectors,
                           "Percent": bmi_percent,
                           "Base": "S&P BMI"
                           })
    # DataFrame for Buffett Portfolio Distribution

    buf_df = pd.DataFrame({"Sector": sectors,
                           "Percent": buffett_pct,
                           "Base": "Buffett BMI"
                           })

    self_df = df.groupby('Sector').sum()
    self_df = self_df.reset_index()[['Sector', 'Market Value']]
    self_df['Percent'] = (self_df['Market Value'] /
                          self_df['Market Value'].sum()).round(4)
    self_df = self_df.drop('Market Value', axis=1)
    self_df['Base'] = 'Current'

    concat_df = pd.concat([bmi_df, buf_df, self_df],
                          join='outer', ignore_index=True)

    df = bmi_df.merge(buf_df)
    df = self_df.merge(bmi_df.merge(buf_df, on='Sector',
                                    how='left'), on='Sector', how='right')
    df = df.rename(columns={'Percent': 'Current',
                            'Percent_x': 'Schwab',
                            'Percent_y': 'Buffett'})
    df = df[['Sector', 'Current', 'Schwab', 'Buffett']]
    df = df.set_index('Sector')
    df = df.fillna(0)
    df = df.T

    sec = df.columns.tolist()
    df_pct = df.style.format(({sec[0]: "{:.2%}",
                               sec[1]: "{:.2%}",
                               sec[2]: "{:.2%}",
                               sec[3]: "{:.2%}",
                               sec[4]: "{:.2%}",
                               sec[5]: "{:.2%}",
                               sec[6]: "{:.2%}",
                               sec[7]: "{:.2%}",
                               sec[8]: "{:.2%}",
                               sec[9]: "{:.2%}",
                               sec[10]: "{:.2%}", }))

    # Figure Generation and Formating
    sns.set(style='white')
    fig = sns.catplot(x="Sector", y='Percent', hue='Base', data=concat_df, kind="bar",
                      height=5, aspect=3, palette=['#D3D3D3', '#C0C0C0', 'C3'], legend_out=True)
    fig.set_xticklabels(rotation=45, fontweight='light', fontsize='large')
    fig.set_xlabels(fontsize='xx-large')
    fig.despine(left='True')
    fig.set_ylabels(fontsize='xx-large')

    plt.title("Portfolio Distribution vs Others", fontdict={'fontsize': 28})

    return df, df_pct


def show_stock_bond_pct(df):
    """
    Show donut chart of Bond vs Stock in comparison Buffett's 70/30 rule
    """
    df = df.groupby("Capitalization").sum()
    df['Percent'] = (
        (df['Market Value'] / df['Market Value'].sum()) * 100).round(2)
    bl, gr, gy = [plt.cm.Blues, plt.cm.Greens, plt.cm.Greys]
    cat = df.index.tolist()
    pct = df['Percent'].tolist()
    labels = np.asarray(["{0} \n {1:.2f}%".format(cat, pct)
                         for cat, pct in zip(cat, pct)])

    BMS_name = ['Bond', 'Stock']
    BMS_pct = [30, 70]
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(BMS_pct, radius=2.5, labels=BMS_name,
                      labeldistance=0.8, colors=[bl(0.2), gr(0.2)])
    plt.setp(mypie, width=0.8, edgecolor='white')
    mypie2, _ = ax.pie(df['Market Value'], radius=2.5 - 0.8, labels=labels,
                       labeldistance=0.5, colors=[bl(0.5), gr(0.5), gy(0.3)])
    plt.setp(mypie2, width=1, edgecolor='white')
    plt.margins(0, 0)
#
    plt.show()
    return


def show_MA(ticker, n=100, start='2010-01-01', ma2=False, n2=20):
    """
    Input Ticker, and n (respective moving average days)
    Preset n = 100 days
           starting date = 2010-01-01
    """
    end = dt.datetime.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, end=end)
    close_px = data['Adj Close']
    mavg = close_px.rolling(window=n).mean()
    plt.figure(figsize=(9, 8))
    style.use('ggplot')
    close_px.plot(label='AAPL')
    mavg.plot(label='MA' + str(n))

    if (ma2 == True):
        mav2 = close_px.rolling(window=n2).mean()
        mav2.plot(label='MA' + str(n2), color='green', style='--')
    else:
        pass

    plt.legend()

    return


def show_heatmap(df):
    """
    Input DataFrame
    Return the heat map for the portfolio price change and percent change
    """

    """
    # Adding empty block to ensure reshape work

    symbol = (np.append((np.asarray(df['Symbol'])),'_').reshape(5,6))
    #create a reshaped array of percent returns that matches the desired shape of the heatmap
    pricechange = (np.append((np.asarray(df['Price_Chg'])),0).reshape(5,6))
    perchange = (np.append((np.asarray(df['Pct_Chg'])),0).reshape(5,6))
    """
    # create a reshaped array of ticker symbols that matches the desired shape of the heatmap
    symbol = (np.asarray(df['Symbol']).reshape(5, 6))
    pricechange = (np.asarray(df['Price_Chg']).reshape(5, 6))
    perchange = (np.asarray(df['Pct_Chg']).reshape(5, 6))

    # create a new array of the same shape as desired, combining the relevant ticker symbol
    # and percentage return data
    labels = (np.asarray(["{0} \n ${1:.2f} \n {2:.2f}%".format(symbol, pricechange, perchange)
                          for symbol, pricechange, perchange in zip(symbol.flatten(),
                                                                    pricechange.flatten(),
                                                                    perchange.flatten())])).reshape(5, 6)

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set(font_scale=1.2)
    plt.title('Portfolio Heatmap for Price and Percent Change', fontsize=18)
    ax.title.set_position([0.5, 1.05])
    ax.set_xticks([])
    sns.heatmap(perchange, annot=labels, fmt="", center=0,  linewidths=0.6, linecolor='white', square=True,
                xticklabels=False, yticklabels=False, cmap='RdYlGn', vmin=-5, vmax=5, ax=ax)
    ax.set_ylim(0, 5)
    ax.invert_yaxis()

    plt.show()
    return

def portfolio_performance(df):

    print()
    portfolio_comparison(df)
    print()
    print('#########################################################')
    print()
    show_stock_bond_pct(df)
    print()
    print('#########################################################')
    print()
    show_heatmap(df)
    print()
    print('#########################################################')
    print()

    return


def check_dividend_growth(ticker, years=10):
    """
    Check dividend growth for the last ten years (preset)
    Input ticker and years (optional)
    Return Figure and DataFrame
    """

    data = yf.Ticker(ticker)
    df = pd.DataFrame(data.dividends)
    df.index = pd.to_datetime(df.index, format="%Y%m").to_period('M')
    periods = dt.datetime.now() - relativedelta(years=years)
    fig = df.loc[periods:].plot.line(
        title=str(ticker) + " - " + str(years) + "-Year Dividend Growth", grid=False)
    fig.set_facecolor('white')
    return df



# Old Way (Archived)
"""
def update_price_values(df):

#     Input a DataFrame
#     Update Tickers' Price and Market Values

    today = dt.datetime.today().strftime("%Y-%m-%d")
    close_price = []
    tickers = df['Symbol'].tolist()

    for ticker in tickers:
        data = yf.download(ticker,today)
        close_price.append([ticker, data.iloc[-1]['Close'].round(2)])
    price_today = pd.DataFrame(close_price,columns=['Symbol','Price'])

    df.sort_values(by='Symbol')
    price_today.sort_values(by='Symbol')
    df.update(price_today)
    df['Market Value'] = df['Quantity']* df['Price']

    return df

def show_cap_bond_distribution(df):
    df = df.groupby("Capitalization").sum()
    df['Percent'] = ((df['Market Value']/df['Market Value'].sum())*100).round(2)
    bl, gr, gy = [plt.cm.Blues, plt.cm.Greens, plt.cm.Greys]
    cat = df.index.tolist()
    pct = df['Percent'].tolist()
    labels = np.asarray(["{0} \n {1:.2f}%".format(cat, pct)
                      for cat, pct in zip(cat, pct)])
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(df['Market Value'], radius=2.5-0.8,labels = labels, labeldistance=0.5, colors= [bl(0.5),gr(0.5), gy(0.3)])
    plt.setp( mypie, width=1, edgecolor='white')
    plt.margins(0,0)

    plt.show()
    return
"""
