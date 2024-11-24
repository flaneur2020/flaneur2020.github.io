import baostock as bs
import pandas as pd
import os
from datetime import datetime, timedelta

def download_hs300_index_data(start_date, end_date):
    """Download HS300 index data between start_date and end_date"""
    rs = bs.query_history_k_data_plus("sh.000300",
        "date,open,high,low,close,volume,amount",
        start_date=start_date, 
        end_date=end_date,
        frequency="d",
        adjustflag="3")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    return data_list, rs.fields

def get_hs300_index():
    """
    Download HS300 index data and save to CSV
    """
    lg = bs.login()
    if lg.error_code != '0':
        print(f'Login failed: {lg.error_msg}')
        return
    
    print('Login successful!')

    file_name = 'hs300_index.csv'
    
    try:
        # Download last 5 years of data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        data_list, fields = download_hs300_index_data(start_date, end_date)
        
        if data_list:
            result = pd.DataFrame(data_list, columns=fields)
            
            # Save to file
            result.to_csv(file_name, encoding='utf-8', index=False)
                
            print(f'\nData saved successfully to: {os.path.abspath(file_name)}')
            print(f'Total records: {len(result)}')
        else:
            print('No data retrieved!')
            
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        
    finally:
        bs.logout()
        print('Logged out from baostock')

if __name__ == "__main__":
    get_hs300_index()
