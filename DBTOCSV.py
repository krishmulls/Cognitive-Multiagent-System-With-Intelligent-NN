import pandas as pd
import pyodbc

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-QK788MA;'
                      'Database=ProjectTest;'
                      'Trusted_Connection=yes;')

sql_query = pd.read_sql_query(''' 
                              select * from ProjectTest.dbo.NNMM_CSV
                              '''
                              ,conn) # here, the 'conn' is the variable that contains your database connection information from step 2

df = pd.DataFrame(sql_query)
df.to_csv (r'D:\Workspace\VS\KnowledgeBase\Data.csv', index = False)

