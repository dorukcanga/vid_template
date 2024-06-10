import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def read_all_fin_data(data_path):
    
    with open(data_path+'quarter_ltm_dfs_dict.pickle', 'rb') as handle:
        quarter_ltm_dfs_dict = pickle.load(handle)

    with open(data_path+'subsector_quarter_ltm_dfs.pickle', 'rb') as handle:
        subsector_quarter_ltm_dfs = pickle.load(handle)

    with open(data_path+'midsector_quarter_ltm_dfs.pickle', 'rb') as handle:
        midsector_quarter_ltm_dfs = pickle.load(handle)

    with open(data_path+'mainsector_quarter_ltm_dfs.pickle', 'rb') as handle:
        mainsector_quarter_ltm_dfs = pickle.load(handle)
        
    return quarter_ltm_dfs_dict, subsector_quarter_ltm_dfs, midsector_quarter_ltm_dfs, mainsector_quarter_ltm_dfs

@st.cache_data
def read_sector_info(data_path):
    
    bist_sectors_final = pd.read_excel(data_path+'bist_sectors_final.xlsx', index_col=0)
    bist_sectors_final.dropna(inplace=True)
    
    return bist_sectors_final

@st.cache_data
def filter_ticker_infos(ticker, quarter_ltm_dfs_dict, bist_sectors_final, subsector_quarter_ltm_dfs, midsector_quarter_ltm_dfs, mainsector_quarter_ltm_dfs):
    df = quarter_ltm_dfs_dict[ticker]
    df = df[df['Satış Gelirleri'].isna() == False]

    ticker_sector_sub = bist_sectors_final[bist_sectors_final.ticker == ticker].sector_class_sub.values[0]
    ticker_sector_mid = bist_sectors_final[bist_sectors_final.ticker == ticker].sector_class_mid.values[0]
    ticker_sector_main = bist_sectors_final[bist_sectors_final.ticker == ticker].sector_class_main.values[0]

    sectors =  [ticker_sector_sub, ticker_sector_mid, ticker_sector_main]

    ticker_sector_dfs = {
        ticker_sector_sub : (subsector_quarter_ltm_dfs[ticker_sector_sub].loc[df.index,:].copy(), 'firebrick'),
        ticker_sector_mid : (midsector_quarter_ltm_dfs[ticker_sector_mid].loc[df.index,:].copy(), 'green'),
        ticker_sector_main : (mainsector_quarter_ltm_dfs[ticker_sector_main].loc[df.index,:].copy(), 'orange')
    }

    return df, sectors, ticker_sector_dfs


######################################################################################################################################################

def return_overview_df(df, chart_type):    
    current_ratios = {
        'Margins' : {
            'Gross Profit Margin' : df['grossprofit_margin'].values[-1],
            'EBITDA Margin' : df['ebitda_margin'].values[-1],
            'EBIT Margin' : df['ebit_margin'].values[-1],
            'Net Profit Margin' : df['netprofit_margin'].values[-1],
            'OCF Margin' : df['ocf_margin'].values[-1],
            'FCF Margin' : df['fcf_margin'].values[-1]
            },
        'Growth' : {
            'Revenue Growth 1Q' : df['rev_growth_1q'].values[-1],
            'EBITDA Growth 1Q' : df['ebitda_growth_1q'].values[-1],
            'EBIT Growth 1Q' : df['ebit_growth_1q'].values[-1],
            'Net Profit Growth 1Q' : df['netprofit_growth_1y'].values[-1],
            'Revenue Growth 1Y' : df['rev_growth_1y'].values[-1],
            'EBITDA Growth 1Y' : df['ebitda_growth_1y'].values[-1],
            'EBIT Growth 1Y' : df['ebit_growth_1y'].values[-1],
            'Net Profit Growth 1Y' : df['netprofit_growth_1y'].values[-1],
            'Revenue CAGR 3Y' : df['rev_cagr_3y'].values[-1],
            'EBITDA CAGR 3Y' : df['ebitda_cagr_3y'].values[-1],
            'EBIT CAGR 3Y' : df['ebit_cagr_3y'].values[-1],
            'Net Profit CAGR 3Y' : df['netprofit_cagr_3y'].values[-1]
            },
        'Profitability' : {
        'Return on Asset' : df['roa'].values[-1],
        'Return on Equity' : df['roe'].values[-1],
        'Return on Invested Capital' : df['roic'].values[-1]
            },
        'Debt' : {
        'Debt to Equity' : df['debt_to_equity'].values[-1],
        'Net Debt to Ebitda' : df['netdebt_to_ebitda'].values[-1],
        'Interest on Coverage Ratio' : df['int_cov_ratio'].values[-1],
        'FCF to Debt' : df['fcf_to_debt'].values[-1]
            },
        'Liquidity' : {
        'Current Ratio' : df['current_ratio'].values[-1],
        'Cash Ratio' : df['cash_ratio'].values[-1],
        'Working Capital Turnover Ratio' : df['wc_turnover_ratio'].values[-1]
            }
    }

    last_quarter_ratios = {
        'Margins' : {
            'Gross Profit Margin' : df['grossprofit_margin'].values[-2],
            'EBITDA Margin' : df['ebitda_margin'].values[-2],
            'EBIT Margin' : df['ebit_margin'].values[-2],
            'Net Profit Margin' : df['netprofit_margin'].values[-2],
            'OCF Margin' : df['ocf_margin'].values[-2],
            'FCF Margin' : df['fcf_margin'].values[-2]
            },
        'Growth' : {
            'Revenue Growth 1Q' : df['rev_growth_1q'].values[-2],
            'EBITDA Growth 1Q' : df['ebitda_growth_1q'].values[-2],
            'EBIT Growth 1Q' : df['ebit_growth_1q'].values[-2],
            'Net Profit Growth 1Q' : df['netprofit_growth_1y'].values[-2],
            'Revenue Growth 1Y' : df['rev_growth_1y'].values[-2],
            'EBITDA Growth 1Y' : df['ebitda_growth_1y'].values[-2],
            'EBIT Growth 1Y' : df['ebit_growth_1y'].values[-2],
            'Net Profit Growth 1Y' : df['netprofit_growth_1y'].values[-2],
            'Revenue CAGR 3Y' : df['rev_cagr_3y'].values[-2],
            'EBITDA CAGR 3Y' : df['ebitda_cagr_3y'].values[-2],
            'EBIT CAGR 3Y' : df['ebit_cagr_3y'].values[-2],
            'Net Profit CAGR 3Y' : df['netprofit_cagr_3y'].values[-2]
            },
        'Profitability' : {
        'Return on Asset' : df['roa'].values[-2],
        'Return on Equity' : df['roe'].values[-2],
        'Return on Invested Capital' : df['roic'].values[-2]
            },
        'Debt' : {
        'Debt to Equity' : df['debt_to_equity'].values[-2],
        'Net Debt to Ebitda' : df['netdebt_to_ebitda'].values[-2],
        'Interest on Coverage Ratio' : df['int_cov_ratio'].values[-2],
        'FCF to Debt' : df['fcf_to_debt'].values[-2]
            },
        'Liquidity' : {
        'Current Ratio' : df['current_ratio'].values[-2],
        'Cash Ratio' : df['cash_ratio'].values[-2],
        'Working Capital Turnover Ratio' : df['wc_turnover_ratio'].values[-2]
            }
    }

    last_year_ratios = {
        'Margins' : {
            'Gross Profit Margin' : df['grossprofit_margin'].values[-5],
            'EBITDA Margin' : df['ebitda_margin'].values[-5],
            'EBIT Margin' : df['ebit_margin'].values[-5],
            'Net Profit Margin' : df['netprofit_margin'].values[-5],
            'OCF Margin' : df['ocf_margin'].values[-5],
            'FCF Margin' : df['fcf_margin'].values[-5]
            },
        'Growth' : {
            'Revenue Growth 1Q' : df['rev_growth_1q'].values[-5],
            'EBITDA Growth 1Q' : df['ebitda_growth_1q'].values[-5],
            'EBIT Growth 1Q' : df['ebit_growth_1q'].values[-5],
            'Net Profit Growth 1Q' : df['netprofit_growth_1y'].values[-5],
            'Revenue Growth 1Y' : df['rev_growth_1y'].values[-5],
            'EBITDA Growth 1Y' : df['ebitda_growth_1y'].values[-5],
            'EBIT Growth 1Y' : df['ebit_growth_1y'].values[-5],
            'Net Profit Growth 1Y' : df['netprofit_growth_1y'].values[-5],
            'Revenue CAGR 3Y' : df['rev_cagr_3y'].values[-5],
            'EBITDA CAGR 3Y' : df['ebitda_cagr_3y'].values[-5],
            'EBIT CAGR 3Y' : df['ebit_cagr_3y'].values[-5],
            'Net Profit CAGR 3Y' : df['netprofit_cagr_3y'].values[-5]
            },
        'Profitability' : {
        'Return on Asset' : df['roa'].values[-5],
        'Return on Equity' : df['roe'].values[-5],
        'Return on Invested Capital' : df['roic'].values[-5]
            },
        'Debt' : {
        'Debt to Equity' : df['debt_to_equity'].values[-5],
        'Net Debt to Ebitda' : df['netdebt_to_ebitda'].values[-5],
        'Interest on Coverage Ratio' : df['int_cov_ratio'].values[-5],
        'FCF to Debt' : df['fcf_to_debt'].values[-5]
            },
        'Liquidity' : {
        'Current Ratio' : df['current_ratio'].values[-5],
        'Cash Ratio' : df['cash_ratio'].values[-5],
        'Working Capital Turnover Ratio' : df['wc_turnover_ratio'].values[-5]
            }
    }

    current_ratios['Dupont Analysis'] = df['dupont'].values[-1]
    last_quarter_ratios['Dupont Analysis'] = df['dupont'].values[-2]
    last_year_ratios['Dupont Analysis'] = df['dupont'].values[-5]


    last_period1, last_period2, last_period4 = df.index[-1], df.index[-2], df.index[-5]

    results_df1 = pd.DataFrame(current_ratios[chart_type].values(), index=current_ratios[chart_type].keys(), columns=[last_period1])
    results_df2 = pd.DataFrame(last_quarter_ratios[chart_type].values(), index=last_quarter_ratios[chart_type].keys(), columns=[last_period2])
    results_df4 = pd.DataFrame(last_year_ratios[chart_type].values(), index=last_year_ratios[chart_type].keys(), columns=[last_period4])

    results_df = results_df1.join(results_df2).join(results_df4)

    results_df['Change QoQ'] = results_df[last_period1] - results_df[last_period2]
    results_df['Change YoY'] = results_df[last_period1] - results_df[last_period4]
    results_df = results_df[[last_period1, last_period2, "Change QoQ", last_period4, "Change YoY"]]

    return results_df

######################################################################################################################################################

def return_selected_labels(labels, default_labels, key=None):
    selected_labels = st.multiselect(label="Select Ratios",
                                       options=labels,
                                       default=default_labels,
                                       key=key
                                     )
    return selected_labels


def return_growth_period(chart_type, key=None):
    
    if chart_type == 'Growth':
        suffix_name = st.selectbox(label="Select Growth Period",
                                    options=tuple(['1 Quarter','1 Year', '3 Year', '5 Year']),
                                    index=0,
                                    key=key
                                 )

        suffix = suffix_name[0]+'q' if suffix_name == '1 Quarter' else suffix_name[0]+'y'
        growth_txt = 'growth' if suffix in ['1q', '1y'] else 'cagr'

    else:
        suffix = '1q'
        growth_txt = 'growth' if suffix in ['1q', '1y'] else 'cagr'
        
    return suffix, growth_txt


labels_dict = {
    'Margins' : [['Gross Profit Margin', 'EBITDA Margin', 'EBIT Margin', 'Net Profit Margin', 'OCF Margin', 'FCF Margin'],
                 ['Gross Profit Margin','EBIT Margin','OCF Margin']],
    'Growth' : [['Revenue Growth', 'EBITDA Growth', 'EBIT Growth', 'Net Profit Growth'],
                 ['Revenue Growth','EBIT Growth']],
    'Profitability' : [['Return on Asset', 'Return on Equity', 'Return on Invested Capital'],
                 ['Return on Equity', 'Return on Invested Capital']],
    'Debt Ratios' : [['Debt to Equity', 'Net Debt to Ebitda', 'Interest on Coverage Ratio', 'FCF to Debt'],
                 ['Debt to Equity', 'Net Debt to Ebitda']],
    'Liquidity Ratios' : [['Current Ratio', 'Cash Ratio', 'Working Capital Turnover Ratio'],
                 ['Current Ratio', 'Cash Ratio']]
}

def return_sectors(sectors, key):
    
    if sectors[0] == sectors[1]:
        sectors2 = sectors[:1].copy()
    elif sectors[1] == sectors[2]:
        sectors2 = sectors[:2].copy()
    else:
        sectors2 = sectors.copy()

    selected_sectors = st.multiselect(label="Select Sectors",
                                   options=sectors2,
                                   default=sectors2,
                                   key=key
                                 )
    
    return selected_sectors

def return_selected_label(labels, key=None):
    label2 = st.selectbox(label="Select Ratio",
                                   options=labels,
                                   index=0,
                                   key=key
                                     )
    return label2

######################################################################################################################################################

def get_hist_ratios_fig(df, ticker, suffix, chart_type, growth_txt, labels):

    
    columns_dict = {
        'Gross Profit Margin' : ('grossprofit_margin', 'royalblue'),
        'EBITDA Margin' : ('ebitda_margin', 'firebrick'),
        'EBIT Margin' : ('ebit_margin', 'green'),
        'Net Profit Margin' : ('netprofit_margin', 'magenta'),
        'OCF Margin' : ('ocf_margin', 'orange'),
        'FCF Margin' : ('fcf_margin', 'black'),
        'Revenue Growth' : ('rev_'+growth_txt+'_'+suffix, 'royalblue'),
        'EBITDA Growth' : ('ebitda_'+growth_txt+'_'+suffix, 'firebrick'),
        'EBIT Growth' : ('ebit_'+growth_txt+'_'+suffix, 'green'),
        'Net Profit Growth' : ('netprofit_'+growth_txt+'_'+suffix, 'purple'),
        'Return on Asset' : ('roa', 'royalblue'),
        'Return on Equity' : ('roe', 'firebrick'),
        'Return on Invested Capital' : ('roic', 'green'),
        'Debt to Equity' : ('debt_to_equity', 'royalblue'),
        'Net Debt to Ebitda' : ('netdebt_to_ebitda', 'firebrick'),
        'Interest on Coverage Ratio' : ('int_cov_ratio', 'green'),
        'FCF to Debt' : ('fcf_to_debt', 'purple'),
        'Current Ratio' : ('current_ratio', 'royalblue'),
        'Cash Ratio' : ('cash_ratio', 'firebrick'),
        'Working Capital Turnover Ratio' : ('wc_turnover_ratio', 'green'),
    }

    columns = [v[0] for k,v in columns_dict.items() if k in labels]
    colors = [v[1] for k,v in columns_dict.items() if k in labels]

    fig = go.Figure()

    annotations = []
    for col, color, label in zip(columns, colors, labels):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=label,
                                 line=dict(color=color, width=2)))

        percentage_string = f'{df[col].values[-1]:.0%}' if chart_type in ['Margins','Growth', 'Profitability'] else f'{df[col].values[-1]:.2f}'

        annotations.append(dict(xref='paper', x=1, y=df[col].values[-1],
                                xanchor='left', yanchor='middle',
                                text=percentage_string,
                                font=dict(family='Arial', size=14, color=color),
                                showarrow=False))

    # Updating layout to enhance the style
    #y_tickformat = ".0%" if chart_type in ['Margins','Growth', 'Profitability'] else '".2f"
    yaxis_dict = dict(tickformat=".0%") if chart_type in ['Margins','Growth', 'Profitability'] else dict(tickformat=".2f")
    fig.update_layout(
        title=dict(text=ticker+' '+chart_type, x=0.5, xanchor='auto'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
        ),
        yaxis=yaxis_dict,
        xaxis=dict(
            showline=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
            ),
        ),
        annotations=annotations,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    hovertemplate = '%{y:.0%}' if chart_type in ['Margins','Growth', 'Profitability'] else '%{y:.2f}'
    # Updating data points to show percentages
    fig.update_traces(
        hovertemplate=hovertemplate
    )
    
    return fig


def get_hist_ratios_sector_fig(df, ticker_sector_dfs, ticker, sectors, chart_type, label, suffix, growth_txt):
    
    
    columns_dict = {
        'Gross Profit Margin' : 'grossprofit_margin',
        'EBITDA Margin' : 'ebitda_margin',
        'EBIT Margin' : 'ebit_margin',
        'Net Profit Margin' : 'netprofit_margin',
        'OCF Margin' : 'ocf_margin',
        'FCF Margin' : 'fcf_margin',
        'Revenue Growth' : 'rev_'+growth_txt+'_'+suffix,
        'EBITDA Growth' : 'ebitda_'+growth_txt+'_'+suffix,
        'EBIT Growth' : 'ebit_'+growth_txt+'_'+suffix,
        'Net Profit Growth' : 'netprofit_'+growth_txt+'_'+suffix,
        'Return on Asset' : 'roa',
        'Return on Equity' : 'roe',
        'Return on Invested Capital' : 'roic',
        'Debt to Equity' : 'debt_to_equity',
        'Net Debt to Ebitda' : 'netdebt_to_ebitda',
        'Interest on Coverage Ratio' : 'int_cov_ratio',
        'FCF to Debt' : 'fcf_to_debt',
        'Current Ratio' : 'current_ratio',
        'Cash Ratio' : 'cash_ratio',
        'Working Capital Turnover Ratio' : 'wc_turnover_ratio'
    }

    col = columns_dict[label]
    
    fig = go.Figure()

    annotations = []
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=ticker,
                             line=dict(color='royalblue', width=2)))

    percentage_string = f'{df[col].values[-1]:.0%}' if chart_type in ['Margins','Growth', 'Profitability'] else f'{df[col].values[-1]:.2f}'
    annotations.append(dict(xref='paper', x=1, y=df[col].values[-1],
                            xanchor='left', yanchor='middle',
                            text=percentage_string,
                            font=dict(family='Arial', size=14, color='royalblue'),
                            showarrow=False))

    for sector in sectors:
        value = ticker_sector_dfs[sector]

        fig.add_trace(go.Scatter(x=value[0].index, y=value[0][col], mode='lines', name=sector,
                                 line=dict(color=value[1], width=2)))

        annotations.append(dict(xref='paper', x=1, y=value[0][col].values[-1],
                                xanchor='left', yanchor='middle',
                                text=f'{value[0][col].values[-1]:.0%}',
                                font=dict(family='Arial', size=14, color=value[1]),
                                showarrow=False))

    # Updating layout to enhance the style
    fig.update_layout(
        title=dict(text=label+' for '+ticker+' vs Sectors', x=0.5, xanchor='auto'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
        ),
        yaxis=dict(
            tickformat=".0%"
        ),
        xaxis=dict(
            showline=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
            ),
        ),
        annotations=annotations,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Updating data points to show percentages
    hovertemplate = '%{y:.0%}' if chart_type in ['Margins','Growth', 'Profitability'] else '%{y:.2f}'
    fig.update_traces(
        hovertemplate=hovertemplate
    )
    
    return fig
