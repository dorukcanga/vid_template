import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from helpers import (read_all_fin_data, read_sector_info, return_overview_df, get_hist_ratios_fig, get_hist_ratios_sector_fig, return_selected_labels, return_growth_period, labels_dict, return_sectors, return_selected_label)


st.set_page_config(page_title="Financial Statement Analysis", layout="wide")

st.title("VID Template")
st.sidebar.title('Choose Ticker')

data_path = "/Users/dorukcanga/Desktop/vid/"

quarter_ltm_dfs_dict, subsector_quarter_ltm_dfs, midsector_quarter_ltm_dfs, mainsector_quarter_ltm_dfs = read_all_fin_data(data_path)

bist_sectors_final = read_sector_info(data_path)
    
ticker_list = list(quarter_ltm_dfs_dict.keys())
ticker_list.sort()

market = st.sidebar.selectbox(
    label="Select Market from List",
    options=tuple(["BIST"]),
    index=0,
    key=0
)

ticker = st.sidebar.selectbox(
    label="Select Ticker from List",
    options=tuple(ticker_list),
    index=ticker_list.index('EREGL'),
    key=1
)

usd_flag = st.sidebar.toggle("Local Currency / USD")

@st.cache_data
def filter_ticker_infos2(ticker):
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


df, sectors, ticker_sector_dfs = filter_ticker_infos2(ticker)
ticker_sector_sub, ticker_sector_mid, ticker_sector_main = sectors[0], sectors[1], sectors[2]

last_period = df.index[-1]


st.divider()

######################################################################################################################################################
st.header('Company Overview for '+ticker)
st.caption('Sector: '+ticker_sector_sub+' | '+ticker_sector_mid+' | '+ticker_sector_main)

overview_df1 = return_overview_df(df=df, chart_type='Margins')
overview_df2 = return_overview_df(df=df, chart_type='Growth')
overview_df3 = return_overview_df(df=df, chart_type='Profitability')
overview_df4 = return_overview_df(df=df, chart_type='Debt')
overview_df5 = return_overview_df(df=df, chart_type='Liquidity')

if sectors[0] == sectors[1]:
    sectors2 = sectors[:1].copy()
elif sectors[1] == sectors[2]:
    sectors2 = sectors[:2].copy()
else:
    sectors2 = sectors.copy()
    
if len(sectors2) <= 1:
    subsec_overview_df1 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Margins')
    subsec_overview_df2 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Growth')
    subsec_overview_df3 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Profitability')
    subsec_overview_df4 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Debt')
    subsec_overview_df5 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Liquidity')
if len(sectors2) <= 2:
    midsec_overview_df1 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Margins')
    midsec_overview_df2 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Growth')
    midsec_overview_df3 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Profitability')
    midsec_overview_df4 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Debt')
    midsec_overview_df5 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Liquidity')
if len(sectors2) <= 3:
    mainsec_overview_df1 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Margins')
    mainsec_overview_df2 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Growth')
    mainsec_overview_df3 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Profitability')
    mainsec_overview_df4 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Debt')
    mainsec_overview_df5 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Liquidity')

subsec_overview_df1 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Margins')
midsec_overview_df1 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Margins')
mainsec_overview_df1 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Margins')

subsec_overview_df2 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Growth')
midsec_overview_df2 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Growth')
mainsec_overview_df2 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Growth')

subsec_overview_df3 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Profitability')
midsec_overview_df3 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Profitability')
mainsec_overview_df3 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Profitability')

subsec_overview_df4 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Debt')
midsec_overview_df4 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Debt')
mainsec_overview_df4 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Debt')

subsec_overview_df5 = return_overview_df(df=ticker_sector_dfs[ticker_sector_sub][0], chart_type='Liquidity')
midsec_overview_df5 = return_overview_df(df=ticker_sector_dfs[ticker_sector_mid][0], chart_type='Liquidity')
mainsec_overview_df5 = return_overview_df(df=ticker_sector_dfs[ticker_sector_main][0], chart_type='Liquidity')

sector_overview_df1 = pd.concat([overview_df1[last_period],
                                 subsec_overview_df1[last_period],
                                 midsec_overview_df1[last_period],
                                 mainsec_overview_df1[last_period]], axis=1)
sector_overview_df1 = sector_overview_df1.iloc[:,:len(sectors2)+1]
sector_overview_df1.columns = [ticker]+sectors2

sector_overview_df2 = pd.concat([overview_df2[last_period],
                                 subsec_overview_df2[last_period],
                                 midsec_overview_df2[last_period],
                                 mainsec_overview_df2[last_period]], axis=1)
sector_overview_df2 = sector_overview_df2.iloc[:,:len(sectors2)+1]
sector_overview_df2.columns = [ticker]+sectors2

sector_overview_df3 = pd.concat([overview_df3[last_period],
                                 subsec_overview_df3[last_period],
                                 midsec_overview_df3[last_period],
                                 mainsec_overview_df3[last_period]], axis=1)
sector_overview_df3 = sector_overview_df3.iloc[:,:len(sectors2)+1]
sector_overview_df3.columns = [ticker]+sectors2

sector_overview_df4 = pd.concat([overview_df4[last_period],
                                 subsec_overview_df4[last_period],
                                 midsec_overview_df4[last_period],
                                 mainsec_overview_df4[last_period]], axis=1)
sector_overview_df4 = sector_overview_df4.iloc[:,:len(sectors2)+1]
sector_overview_df4.columns = [ticker]+sectors2

sector_overview_df5 = pd.concat([overview_df5[last_period],
                                 subsec_overview_df5[last_period],
                                 midsec_overview_df5[last_period],
                                 mainsec_overview_df5[last_period]], axis=1)
sector_overview_df5 = sector_overview_df5.iloc[:,:len(sectors2)+1]
sector_overview_df5.columns = [ticker]+sectors2

def highlight_values(val):
    color = 'red' if val < 0 else 'green'
    return f'color: {color}'

tab_list = ['Summary','Valuation', 'Margins','Growth', 'Profitability', 'Debt', 'Liquidity']
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_list)
with tab1:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(label='P/E', value=np.nan)
        st.metric(label='EV/EBIT', value=np.nan)
    with col2:
        st.metric(label='Gross Margin', value='{:.1%}'.format(df['grossprofit_margin'].values[-1]))
        st.metric(label='EBIT Margin', value='{:.1%}'.format(df['ebit_margin'].values[-1]))
    with col3:
        st.metric(label='Revenue Growth 1Y', value='{:.1%}'.format(df['rev_growth_1y'].values[-1]))
        st.metric(label='Revenue CAGR 3Y', value='{:.1%}'.format(df['rev_cagr_3y'].values[-1]))
    with col4:
        st.metric(label='RoE', value='{:.1%}'.format(df['roe'].values[-1]))
        st.metric(label='ROIC', value='{:.1%}'.format(df['roic'].values[-1]))
    with col5:
        st.metric(label='Debt to Equity', value='{:.2f}'.format(df['debt_to_equity'].values[-1]))
        st.metric(label='Net Debt to Ebitda', value='{:.2f}'.format(df['netdebt_to_ebitda'].values[-1]))
    with col6:
        st.metric(label='Dividend Yield', value=np.nan)
        st.metric(label='Payout Ratio', value='{:.1%}'.format(df['dividend_payout_ratio'].values[-1]))
    

with tab2:
    st.write('Lorem Ipsum')
    
with tab3:
    overview_df_style1 = overview_df1.style.format({i:"{:.1%}" for i in overview_df1.columns})\
                                        .applymap(highlight_values, subset=pd.IndexSlice[:, ['Change QoQ', 'Change YoY']])

    st.dataframe(overview_df_style1, use_container_width=True, height=200)
    st.dataframe(sector_overview_df1.style.format({i:"{:.1%}" for i in sector_overview_df1.columns}), use_container_width=True, height=200)
        
with tab4:
    
    overview_df2_1 = overview_df2.loc[[i for i in overview_df2.index if 'Revenue' in i],:]
    overview_df2_2 = overview_df2.loc[[i for i in overview_df2.index if 'EBITDA' in i],:]
    overview_df2_3 = overview_df2.loc[[i for i in overview_df2.index if 'EBIT ' in i],:]
    overview_df2_4 = overview_df2.loc[[i for i in overview_df2.index if 'Net Profit' in i],:]
    
    overview_df_style2_1 = overview_df2_1.style.format({i:"{:.1%}" for i in overview_df2_1.columns})\
                                        .applymap(highlight_values, subset=pd.IndexSlice[:, ['Change QoQ', 'Change YoY']])
    overview_df_style2_2 = overview_df2_2.style.format({i:"{:.1%}" for i in overview_df2_2.columns})\
                                        .applymap(highlight_values, subset=pd.IndexSlice[:, ['Change QoQ', 'Change YoY']])
    overview_df_style2_3 = overview_df2_3.style.format({i:"{:.1%}" for i in overview_df2_3.columns})\
                                        .applymap(highlight_values, subset=pd.IndexSlice[:, ['Change QoQ', 'Change YoY']])
    overview_df_style2_4 = overview_df2_4.style.format({i:"{:.1%}" for i in overview_df2_4.columns})\
                                        .applymap(highlight_values, subset=pd.IndexSlice[:, ['Change QoQ', 'Change YoY']])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.dataframe(overview_df_style2_1, use_container_width=True, height=200)
    with col2:
        st.dataframe(overview_df_style2_2, use_container_width=True, height=200)
    with col3:
        st.dataframe(overview_df_style2_3, use_container_width=True, height=200)
    with col4:
        st.dataframe(overview_df_style2_4, use_container_width=True, height=200)
        
    st.dataframe(sector_overview_df2.style.format({i:"{:.1%}" for i in sector_overview_df2.columns}), use_container_width=True, height=200)
        
with tab5:

    overview_df_style3 = overview_df3.style.format({i:"{:.1%}" for i in overview_df3.columns})\
                                        .applymap(highlight_values, subset=pd.IndexSlice[:, ['Change QoQ', 'Change YoY']])

    st.dataframe(overview_df_style3, use_container_width=True, height=200)
    st.dataframe(sector_overview_df3.style.format({i:"{:.1%}" for i in sector_overview_df3.columns}), use_container_width=True, height=200)
        
with tab6:
    
    overview_df_style4 = overview_df4.style.format({i:"{:.2f}" for i in overview_df4.columns})\
                                        .applymap(highlight_values, subset=pd.IndexSlice[:, ['Change QoQ', 'Change YoY']])

    st.dataframe(overview_df_style4, use_container_width=True, height=200)
    st.dataframe(sector_overview_df4.style.format({i:"{:.2f}" for i in sector_overview_df4.columns}), use_container_width=True, height=200)
        
with tab7:
    overview_df_style5 = overview_df5.style.format({i:"{:.2f}" for i in overview_df5.columns})\
                                        .applymap(highlight_values, subset=pd.IndexSlice[:, ['Change QoQ', 'Change YoY']])

    st.dataframe(overview_df_style5, use_container_width=True, height=200)
    st.dataframe(sector_overview_df5.style.format({i:"{:.2f}" for i in sector_overview_df5.columns}), use_container_width=True, height=200)


st.divider()

######################################################################################################################################################

with st.expander("See Historical Financial Results for "+ticker):

    st.header('Historical Financial Results for '+ticker)


    tab_list = ['Valuation', 'Margins','Growth', 'Profitability', 'Debt', 'Liquidity']
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)
    with tab1:
        st.write('Lorem Ipsum')

    with tab2:
        chart_type = 'Margins'
        labels = labels_dict[chart_type][0]
        default_labels = labels_dict[chart_type][1]

        selected_labels = return_selected_labels(labels, default_labels, 2)
        suffix, growth_txt = return_growth_period(chart_type)

        fig = get_hist_ratios_fig(df, ticker, suffix, chart_type, growth_txt, selected_labels)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with tab3:
        chart_type = 'Growth'
        labels = labels_dict[chart_type][0]
        default_labels = labels_dict[chart_type][1]

        col1, col2 = st.columns(2)
        with col1:
            selected_labels = return_selected_labels(labels, default_labels, 3)
        with col2:
            suffix, growth_txt = return_growth_period(chart_type, 4)

        fig = get_hist_ratios_fig(df, ticker, suffix, chart_type, growth_txt, selected_labels)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with tab4:
        chart_type = 'Profitability'
        labels = labels_dict[chart_type][0]
        default_labels = labels_dict[chart_type][1]

        selected_labels = return_selected_labels(labels, default_labels, 5)
        suffix, growth_txt = return_growth_period(chart_type)

        fig = get_hist_ratios_fig(df, ticker, suffix, chart_type, growth_txt, selected_labels)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with tab5:
        chart_type = 'Debt Ratios'
        labels = labels_dict[chart_type][0]
        default_labels = labels_dict[chart_type][1]

        selected_labels = return_selected_labels(labels, default_labels, 6)
        suffix, growth_txt = return_growth_period(chart_type)

        fig = get_hist_ratios_fig(df, ticker, suffix, chart_type, growth_txt, selected_labels)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with tab6:
        chart_type = 'Liquidity Ratios'
        labels = labels_dict[chart_type][0]
        default_labels = labels_dict[chart_type][1]

        selected_labels = return_selected_labels(labels, default_labels, 7)
        suffix, growth_txt = return_growth_period(chart_type)

        fig = get_hist_ratios_fig(df, ticker, suffix, chart_type, growth_txt, selected_labels)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)
            

st.divider()

######################################################################################################################################################

with st.expander("See Historical Financial Results for "+ticker+" vs Sector"):
    
    st.header('Historical Financial Results for '+ticker+' vs Sector')


    tab_list = ['Valuation', 'Margins','Growth', 'Profitability', 'Debt', 'Liquidity']
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)
    with tab1:
        st.write('Lorem Ipsum')

    with tab2:
        chart_type = 'Margins'
        labels = labels_dict[chart_type][0]

        col1, col2 = st.columns(2)
        with col1:
            selected_sectors = return_sectors(sectors, 8)

        with col2:
            label2 = return_selected_label(labels, 9)
            suffix, growth_txt = return_growth_period(chart_type)

        fig = get_hist_ratios_sector_fig(df, ticker_sector_dfs, ticker, selected_sectors, chart_type, label2, suffix, growth_txt)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with tab3:
        chart_type = 'Growth'
        labels = labels_dict[chart_type][0]

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_sectors = return_sectors(sectors, 10)
        with col2:
            label2 = return_selected_label(labels, 11)
        with col3:
            suffix, growth_txt = return_growth_period(chart_type, 12)

        fig = get_hist_ratios_sector_fig(df, ticker_sector_dfs, ticker, selected_sectors, chart_type, label2, suffix, growth_txt)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with tab4:
        chart_type = 'Profitability'
        labels = labels_dict[chart_type][0]
        col1, col2 = st.columns(2)
        with col1:
            selected_sectors = return_sectors(sectors, 13)

        with col2:
            label2 = return_selected_label(labels, 14)
            suffix, growth_txt = return_growth_period(chart_type)

        fig = get_hist_ratios_sector_fig(df, ticker_sector_dfs, ticker, selected_sectors, chart_type, label2, suffix, growth_txt)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with tab5:
        chart_type = 'Debt Ratios'
        labels = labels_dict[chart_type][0]
        col1, col2 = st.columns(2)
        with col1:
            selected_sectors = return_sectors(sectors, 15)

        with col2:
            label2 = return_selected_label(labels, 16)
            suffix, growth_txt = return_growth_period(chart_type)

        fig = get_hist_ratios_sector_fig(df, ticker_sector_dfs, ticker, selected_sectors, chart_type, label2, suffix, growth_txt)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with tab6:
        chart_type = 'Liquidity Ratios'
        labels = labels_dict[chart_type][0]
        col1, col2 = st.columns(2)
        with col1:
            selected_sectors = return_sectors(sectors, 17)

        with col2:
            label2 = return_selected_label(labels, 18)
            suffix, growth_txt = return_growth_period(chart_type)

        fig = get_hist_ratios_sector_fig(df, ticker_sector_dfs, ticker, selected_sectors, chart_type, label2, suffix, growth_txt)
        st.plotly_chart(figure_or_data=fig, use_container_width=True)




    
    
    
    
    
    
    
    
    
