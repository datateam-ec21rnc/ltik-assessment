import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from PIL import Image
import base64 
import sys
import os
import importlib.util

# 페이지 레이아웃 설정 (wide 모드)
st.set_page_config(layout="wide")

# ---------------------------------- # 
# 공통 인증 모듈 import
# auth 폴더 경로 설정
auth_path = os.path.join(os.path.dirname(__file__), '..', 'auth', 'auth.py')
spec = importlib.util.spec_from_file_location("auth", auth_path)
auth_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth_module)
check_password = auth_module.check_password

# 비밀번호 확인
if not check_password():
    st.stop()

# ---------------------------------- # 
# 테이블명 매핑 딕셔너리
TABLE_NAME_MAPPING = {
    'Raw1': '로데이터1(작품)',
    'Raw2': '로데이터2(작품)',
    'annual_currency_weights': '적용 환율',
    'evaluation_results': '출판사 성과 등급',
    'gdp_weights': '국가별 GDP 가중치',
    'master_list': '출판사 연번',
    'publisher_classification': '출판사 연간 실적 현황'
}

def get_display_name(table_name):
    """테이블명을 표시용 이름으로 변환"""
    return TABLE_NAME_MAPPING.get(table_name, table_name)

# ---------------------------------- # 

def connect_db():
    """SQLite 데이터베이스 연결"""
    # 사용자 이름 확인
    username = os.getenv('USERNAME') or os.getenv('USER')
    
    if username == 'EC21RNC':
        # 사용자: 파일 위치 기준 경로 사용 (mei)
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'DB', 'publisher_evaluation.db')
        db_path = os.path.normpath(db_path)
    else:
        # 다른 사용자: 작업 디렉토리 기준 경로 사용 (리눅스, Streamlit cloud)
        db_path = "./DB/publisher_evaluation.db"
    
    return sqlite3.connect(db_path)

def get_table_list():
    """데이터베이스의 모든 테이블 목록 조회"""
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    finally:
        conn.close()

def get_table_data(table_name):
    """선택한 테이블의 데이터 조회"""
    conn = connect_db()
    try:
        df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
        return df
    finally:
        conn.close()

def to_excel(df):
    """DataFrame을 Excel 바이트로 변환"""
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# ---------------------------------- # 

st.header("📄 데이터 확인")
st.write("DB의 테이블을 확인하고 엑셀로 다운로드할 수 있습니다.")

# 테이블 목록 조회
try:
    tables = get_table_list()
    
    if not tables:
        st.warning("데이터베이스에 테이블이 없습니다.")
    else:
        # st.success(f"총 {len(tables)}개의 테이블을 찾았습니다.")
        pass
        
        # 기본 선택 테이블 설정
        default_table = 'publisher_classification'
        default_index = tables.index(default_table) if default_table in tables else 0
        
        # 테이블 선택 (매핑된 이름으로 표시)
        selected_table = st.selectbox(
            "조회할 테이블을 선택하세요:",
            tables,
            format_func=lambda x: get_display_name(x),
            key="table_selector",
            index=default_index
        )
        
        if selected_table:
            
            
            # 데이터 조회
            try:
                df = get_table_data(selected_table)
                
                # 테이블 정보 표시 (한 줄)
                display_name = get_display_name(selected_table)
                st.caption(f"ℹ️ 테이블: `{display_name}` | 행 수: {len(df):,} | 열 수: {len(df.columns)}")
                st.divider()
                # 데이터 미리보기
                st.write("📋 **데이터 미리보기**")
                st.dataframe(df, use_container_width=True, height=400)
                
                # 엑셀 다운로드 버튼
                excel_data = to_excel(df)
                display_name = get_display_name(selected_table)
                
                st.download_button(
                    label=f"📥 데이터 다운로드 ({display_name}.xlsx)",
                    data=excel_data,
                    file_name=f"{display_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{selected_table}",
                    use_container_width=True,
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"데이터 조회 중 오류가 발생했습니다: {str(e)}")
                
except sqlite3.Error as e:
    st.error(f"데이터베이스 연결 오류: {str(e)}")
except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.exception(e) 

# 푸터
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div style="font-size: 0.9em; color: #666;">출판사 역량분석 시스템 v2.0</div>
    <div style="font-size: 0.9em; color: #666;">EC21R&C Inc.</div>
</div>
""", unsafe_allow_html=True)
