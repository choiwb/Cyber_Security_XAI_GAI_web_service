


from setting import *

import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkConf
import os
from sklearn.feature_extraction.text import CountVectorizer
from flask import request


def predict_UI_sql_result():

    # Flask Web 연동 시, input 입력 데이터
    raw_data_str = request.form['raw_data_str']
    # cProfile 을 통한 함수 실행 시간 파악하기 위한 테스트 데이터
    # raw_data_str = 'GET /robots.txt HTTP/1.1\r\nUser-Agent: Mozilla/5.0 (compatible; Nmap Scripting Engine)'

    # Local 실행 시, java 경로
    java11_location= 'JAVA PATH !!!!!'
    os.environ['JAVA_HOME'] = java11_location

    conf = pyspark.SparkConf().setAppName('prep_data').setMaster('local')
    sc = pyspark.SparkContext.getOrCreate(conf = conf)

    # pyspark session 정보 확인
    print(SparkConf().getAll())

    # 세션 수행
    session = SparkSession(sc)

    domain_one_row_df = pd.DataFrame(data = [raw_data_str],
                                    columns = ['payload'])

    schema = StructType([StructField("payload", StringType(), True)
                    ])

    # 데이터 프레임 등록
    domain_df = session.createDataFrame(domain_one_row_df, schema=schema)

    # 현재 스키마 정보 확인
    domain_df.printSchema()

    # 데이터 프레임 'table'이라는 이름으로 SQL테이블 생성
    domain_df.createOrReplaceTempView("table") #<=== SparkSQL에 생성된 테이블 이름

    # domain 피처
    sql_query = """

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

            SIZE(SPLIT(IF(ISNULL(payload), '', payload), '[\\.][\\.]'))-1 AS ips_00032_payload_char_double_dot_cnt,

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
    
    # IPS 신규 표준 피처 기반 모델 단, payload 이외 필드 참조하는 피처 3개 제외 
    # (ips_00001_attack_ip_method, ips_00001_count_value ips_00001_count_value)
    new_sql_query = """
    
    SELECT
   
        IF(INT(RLIKE(payload, 'VCAvY2dpLWJpbi9waHA0') )>0
        OR INT(RLIKE(payload, 'L2NnaS1iaW4v') )>0
        OR INT(RLIKE(payload, 'IC9jZ2ktYmlu') )>0
        OR INT(RLIKE(payload, 'UE9TVCAvY2dpLWJpbi9waHA/') )>0
        OR INT(RLIKE(payload, 'VCAvY2dpLWJpbi9w') )>0
        OR INT(RLIKE(payload, 'ZGllKEBtZDU=') )>0
        OR INT(RLIKE(payload, 'L2FueWZvcm0yL3VwZGF0ZS9hbnlmb3JtMi5pbmk=') )>0
        OR INT(RLIKE(payload, 'Ly5iYXNoX2hpc3Rvcnk=') )>0
        OR INT(RLIKE(payload, 'L2V0Yy9wYXNzd2Q=') )>0
        OR INT(RLIKE(payload, 'QUFBQUFBQUFBQQ==') )>0
        OR INT(RLIKE(payload, 'IG1hc3NjYW4vMS4w') )>0
        OR INT(RLIKE(payload, 'd2dldA==') )>0
        OR INT(RLIKE(payload, 'MjB3YWl0Zm9yJTIwZGVsYXklMjAn') )>0
        OR INT(RLIKE(payload, 'V0FJVEZPUiBERUxBWQ==') )>0
        OR INT(RLIKE(payload, 'ZXhlYw==') )>0
        OR INT(RLIKE(payload, 'Tm9uZQ==') )>0
        OR INT(RLIKE(payload, 'OyB3Z2V0') )>0
        OR INT(RLIKE(payload, 'VXNlci1BZ2VudDogRGlyQnVzdGVy') )>0
        OR INT(RLIKE(payload, 'cGhwIGRpZShAbWQ1') )>0
        OR INT(RLIKE(payload, 'JTI4U0VMRUNUJTIw') )>0
                ,1, 0) AS ips_00001_payload_base64,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('select', '(.*?)', 'from')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('select', '(.*?)', 'count')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('select', '(.*?)', 'distinct')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('union', '(.*?)', 'select')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('select', '(.*?)', 'extractvalue', '.*', 'xmltype')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('from', '(.*?)', 'generate', '.*', 'series')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('from', '(.*?)', 'group', '.*', 'by')) )>0
                ,1, 0) AS ips_00001_payload_sql_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('case', '(.*?)', 'when')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('then', '(.*?)', 'else')) )>0
                ,1, 0) AS ips_00001_payload_sql_comb_02,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('waitfor', '(.*?)', 'delay')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('db', '(.*?)', 'sql', '(.*?)', 'server')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('cast', '(.*?)', 'chr')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('like', '(.*?)', 'http/1.')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('upper', '(.*?)', 'xmltype')) )>0
                ,1, 0) AS ips_00001_payload_sql_comb_03,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('script', '(.*?)', 'alert')) )>0
                ,1, 0) AS ips_00001_payload_xss_comb_01,

        IF(INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  CONCAT('wget', '(.*?)', 'ttp')) )>0
        OR INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  CONCAT('chmod', '(.*?)', '777')) )>0
        OR INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  CONCAT('rm', '(.*?)', 'rf')) )>0
        OR INT(RLIKE(SPLIT(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), 'http/1.', 2)[0],  CONCAT('cd', '(.*?)', 'tmp')) )>0
                ,1, 0) AS ips_00001_payload_cmd_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('jndi', '(.*?)', 'dap')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '),  CONCAT('jndi', '(.*?)', 'dns')) )>0
                ,1, 0) AS ips_00001_payload_log4j_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('etc', '(.*?)', 'passwd')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('document', '(.*?)', 'createelement')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('cgi', '(.*?)', 'bin')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('document', '(.*?)', 'forms')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('document', '(.*?)', 'location')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('fckeditor', '(.*?)', 'filemanager')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('manager', '(.*?)', 'html')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('current_config', '(.*?)', 'passwd')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('currentsetting', '(.*?)', 'htm')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('well', '(.*?)', 'known')) )>0
                ,1, 0) AS ips_00001_payload_word_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('bash', '(.*?)', 'history')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('apache', '(.*?)', 'struts')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('document', '(.*?)', 'open')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('backup', '(.*?)', 'sql')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('robots', '(.*?)', 'txt')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('sqlexec', '(.*?)', 'php')) )>0
        OR INT(RLIKE(LOWER(payload), 'htaccess') )>0
        OR INT(RLIKE(LOWER(payload), 'htpasswd') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('cgi', '(.*?)', 'cgi')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('api', '(.*?)', 'ping')) )>0
                ,1, 0) AS ips_00001_payload_word_comb_02,

        IF(INT(RLIKE(LOWER(payload), 'aaaaaaaaaa') )>0
        OR INT(RLIKE(LOWER(payload), 'cacacacaca') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('mozi', '[\\.]')) )>0
        OR INT(RLIKE(LOWER(payload), 'bingbot') )>0
        OR INT(RLIKE(LOWER(payload), 'md5') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('jpg', '(.*?)', 'http', '(.*?)', '1.1')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('count', '(.*?)', 'cgi', '(.*?)', 'http')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('this', '(.*?)', 'program', '(.*?)', 'can')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('sleep', '(.*?)', 'sleep')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('and', '(.*?)', 'sleep')) )>0
        OR INT(RLIKE(LOWER(payload), 'delete'))>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('get', '(.*?)', 'ping')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('msadc', '(.*?)', 'dll', '(.*?)', 'http')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('filename', '(.*?)', 'asp')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('filename', '(.*?)', 'jsp')) )>0
                ,1, 0) AS ips_00001_payload_word_comb_03,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('wp', '(.*?)', 'login')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('wp', '(.*?)', 'content')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('wp', '(.*?)', 'include')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('wp', '(.*?)', 'config')) )>0
                ,1, 0) AS ips_00001_payload_wp_comb_01,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('cmd', '(.*?)', 'open')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('echo', '(.*?)', 'shellshock')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('php', '(.*?)', 'echo')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('admin', '(.*?)', 'php')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload), '\\n|\\r|\\t', ' '), CONCAT('script', '(.*?)', 'setup', '(.*?)', 'php')) )>0
        OR INT(RLIKE(LOWER(payload), 'phpinfo') )>0
        OR INT(RLIKE(LOWER(payload), 'administrator') )>0
        OR INT(RLIKE(LOWER(payload), 'phpmyadmin') )>0
        OR INT(RLIKE(LOWER(payload), 'access') )>0
        OR INT(RLIKE(LOWER(payload), 'eval') )>0
        OR INT(RLIKE(LOWER(payload), 'mdb') )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('wise', '(.*?)', 'survey', '(.*?)', 'admin')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('admin', '(.*?)', 'serv', '(.*?)', 'admpw')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('php', '(.*?)', 'create', '(.*?)', 'function')) )>0
                ,1, 0) AS ips_00001_payload_word_comb_04,

        IF(INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'zgrab')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'nmap')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'dirbuster')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'ahrefsbot')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'baiduspider')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'mj12bot')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'petalbot')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'semrushbot')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'curl/')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'masscan')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'sqlmap')) )>0
        OR INT(RLIKE(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('user', '-', 'agent', '(.*?)', 'urlgrabber', '(.*?)', 'yum')) )>0
                ,1, 0) AS ips_00001_payload_useragent_comb,

        (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('get', '(.*?)', 'http/1.'))) -1)
            + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('post', '(.*?)', 'http/1.'))) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('head', '(.*?)', 'http/1.'))) -1)
        + (SIZE(SPLIT(REGEXP_REPLACE(LOWER(payload),  '\\n|\\r|\\t', ' '), CONCAT('option', '(.*?)', 'http/1.'))) -1)
        AS ips_00001_payload_whitelist
    FROM table
    
    """

    # 예측 데이터 - TF-IDF 피처 생성
    
    # train 셋의 키워드 및 IDF 값 호출
    train_word_idf = pd.read_csv('train_word_idf.csv')
    train_word = list(train_word_idf['word'])
    train_idf = list(train_word_idf['idf'])

    counter = CountVectorizer(lowercase=True, vocabulary = train_word)
    payload_counter = counter.fit_transform(domain_one_row_df['payload']).toarray()

    valid_count_df = pd.DataFrame(payload_counter, columns=counter.get_feature_names_out())
    valid_count_df = valid_count_df.rename(columns = lambda x: 'payload_' + x)

    # pandas data frame 형태 예측 데이터 TF-IDF 피처 생성
    valid_tfidf_df = valid_count_df * train_idf

    # 쿼리 실행하고, 결과 데이터 프레임에 저장
    # output_df = session.sql(sql_query) #<==== 쿼리를 실행하는 부분
    output_df = session.sql(new_sql_query) #<==== 쿼리를 실행하는 부분

    
    sql_result_df = output_df.toPandas()
    sql_result_df = pd.concat([sql_result_df, valid_tfidf_df], axis = 1)

    # CatBoost 모델링 위해 연속형의 경우, str 또는 int 형 변환 필요
    # sql_result_df_result['ips_00014_payload_logscaled_length_value'] = sql_result_df_result['ips_00014_payload_logscaled_length_value'].astype(int)
    # LightGBM 및 XGBoost 계열은 연속형 피처도 학습 가능하므로, 형봔한 X
    
    print('전처리 데이터 크기: ', sql_result_df.shape)
    print('전처리 데이터 샘플: ', sql_result_df)


    return sql_result_df
