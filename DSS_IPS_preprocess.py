



from setting import *

import pandas as pd, numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *

import os

from flask import request, Flask


app = Flask(__name__)
app.route('/predict_UI_sql_result', methods = ['POST'])

def predict_UI_sql_result():

    raw_data_str = request.form['raw_data_str']
    # raw_data_str = 'GET /robots.txt HTTP/1.1\r\nUser-Agent: Mozilla/5.0 (compatible; Nmap Scripting Engine'

    java11_location= '/opt/homebrew/opt/openjdk@11'
    os.environ['JAVA_HOME'] = java11_location


    conf = pyspark.SparkConf().setAppName('prep_data').setMaster('local')
    # sc = pyspark.SparkContext(conf=conf)
    sc = pyspark.SparkContext.getOrCreate(conf = conf)

    # 세션 수행
    session = SparkSession(sc)

    payload = raw_data_str

    domain_one_row_df = pd.DataFrame(data = [[payload]],
                                    columns = ['payload'])



    schema = StructType([StructField("payload", StringType(), True)
                    ])

    # 데이터 프레임 등록
    domain_df = session.createDataFrame(domain_one_row_df, schema=schema)

    # 현재 스키마 정보 확인
    domain_df.printSchema()

    # 데이터 프레임 'table'이라는 이름으로 SQL테이블 생성
    domain_df.createOrReplaceTempView("table") #<=== SparkSQL에 생성된 테이블 이름


    query_1 = """

            SELECT 

            CHAR_LENGTH(IF(ISNULL(payload) OR (LOWER(payload) IN ("", " ", "-", "null", "nan")), "", payload)) AS ips_00013_payload_length_value,

            IF(CHAR_LENGTH(IF(ISNULL(payload) OR (LOWER(payload) IN ("", " ", "-", "null", "nan")), "", payload))<1, 0, LN(CHAR_LENGTH(IF(ISNULL(payload) OR (LOWER(payload) IN ("", " ", "-", "null", "nan")), "", payload)))) AS ips_00014_payload_logscaled_length_value,

            IF(INSTR(LOWER(IF(ISNULL(payload), "", payload)), "manager")>0, 1, 0) AS ips_00015_payload_sys_manager_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), "", payload)), "console")>0, 1, 0) AS ips_00016_payload_sys_console_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), "", payload)), "admin")>0, 1, 0) AS ips_00017_payload_sys_admin_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), "", payload)), "setup")>0, 1, 0) AS ips_00018_payload_sys_setup_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), "", payload)), "config")>0, 1, 0) AS ips_00019_payload_sys_config_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), "", payload)), "server")>0, 1, 0) AS ips_00020_payload_sys_server_flag,

            SIZE(SPLIT(IF(ISNULL(payload), "", payload), "[\']"))-1 AS ips_00021_payload_char_single_quotation_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\"]'))-1 AS ips_00022_payload_char_double_quotation_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\=]')) - SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\&]')) AS ips_00023_payload_char_equal_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\+]'))-1 AS ips_00024_payload_char_plus_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\*]'))-1 AS ips_00025_payload_char_star_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\/]'))-1 AS ips_00026_payload_char_slush_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\<]'))-1 AS ips_00027_payload_char_lt_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\@]'))-1 AS ips_00028_payload_char_at_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\(]'))-1 AS ips_00029_payload_char_parent_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\{]'))-1 AS ips_00030_payload_char_bracket_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\$]'))-1 AS ips_00031_payload_char_dollar_cnt,

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\.][\\.]'))-1 AS ips_00032_payload_char_double_dot_cnt

            FROM table

    """


    query_2 = """

            SELECT

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT(ch, 'and', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00033_payload_sql_and_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT(ch, 'or', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00034_payload_sql_or_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT(ch, 'select', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00035_payload_sql_select_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT(ch, 'from', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00036_payload_sql_from_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), CONCAT('cast', CHR(40)))>0, 1, 0) AS ips_00037_payload_sql_cast_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('union', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00038_payload_sql_union_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), CONCAT('eval', CHR(40)))>0, 1, 0) AS ips_00039_payload_sql_eval_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), CONCAT('char', CHR(40)))>0, 1, 0) AS ips_00040_payload_sql_char_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), CONCAT('base64', CHR(40)))>0, 1, 0) AS ips_00041_payload_sql_base64_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('declare', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00042_payload_sql_declare_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), 'alert')>0, 1, 0) AS ips_00043_payload_xss_alert_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), 'script')>0, 1, 0) AS ips_00044_payload_xss_script_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), 'document')>0, 1, 0) AS ips_00045_payload_xss_document_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), 'onmouseover')>0, 1, 0) AS ips_00046_payload_xss_onmouseover_flag,

            IF(INSTR(LOWER(IF(ISNULL(payload), '', payload)), 'onload')>0, 1, 0) AS ips_00047_payload_xss_onload_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('cmd', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00048_payload_cmd_cmd_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('run', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00049_payload_cmd_run_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('config', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00050_payload_cmd_config_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('ls', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00051_payload_cmd_ls_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('mkdir', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00052_payload_cmd_mkdir_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('netstat', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00053_payload_cmd_netstat_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('ftp', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00054_payload_cmd_ftp_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('cat', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00055_payload_cmd_cat_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('dir', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00056_payload_cmd_dir_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('wget', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00057_payload_cmd_wget_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('echo', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00058_payload_cmd_echo_flag,

            IF(AGGREGATE(TRANSFORM(TRANSFORM(ARRAY(' ', CONCAT(CHR(37), '20'), CHR(43)), ch -> CONCAT('rm', ch)), word -> INT(INSTR(LOWER(IF(ISNULL(payload), '', payload)), word))), 0, (x1, x2) -> x1+x2)>0, 1, 0) AS ips_00059_payload_cmd_rm_flag


            FROM table


    """


    # 쿼리 실행하고, 결과 데이터 프레임에 저장
    output_df = session.sql(query_1) #<==== 쿼리를 실행하는 부분
    output_df_2 = session.sql(query_2) #<==== 쿼리를 실행하는 부분


    # 데이터 확인
    # print(output_df.show(20))
    # print(output_df_2.show(20))


    sql_result_df = output_df.toPandas()
    sql_result_df_2 = output_df_2.toPandas()

    sql_result_df_result = pd.concat([sql_result_df, sql_result_df_2], axis = 1)

    sql_result_df_result['ips_00014_payload_logscaled_length_value'] = sql_result_df_result['ips_00014_payload_logscaled_length_value'].astype(int)

    print('전처리 데이터 크기: ', sql_result_df_result.shape)
    print('전처리 데이터 샘플: ', sql_result_df_result)

    '''
    # DS 서버상에, 사용자가 입력한 payload 데이터, 예측 결과, 예측 확률을 DB 상에 저장 필요!!!!!!!!!!!!!!!!!!!!!!!!!
    user_payload_data_log_save_DB = user_payload_data_log_save_DB.APPEND(sql_result_df_result)
    '''


    return sql_result_df_result


if __name__ == '__main__':
    app.run(host = SERVER_IP, port = PORT)