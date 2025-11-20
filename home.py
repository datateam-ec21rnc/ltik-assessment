'''
https://polar-crystal-78f.notion.site/DB-2ab2474e3bd9806ca137cbd4792bd59b
'''
import streamlit as st
import pandas as pd
import sqlite3
import numpy as np 
from PIL import Image
import base64 

YEAR = 2025 

logo = Image.open('./assets/logo1.jpg')  # 또는 'assets/logo.png'
def get_base64_image(image_path):
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image('./assets/logo1.jpg')

# 공통 인증 모듈 import
import sys
import os
import importlib.util

# auth 폴더 경로 설정
auth_path = os.path.join(os.path.dirname(__file__), 'auth', 'auth.py')
spec = importlib.util.spec_from_file_location("auth", auth_path)
auth_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth_module)
check_password = auth_module.check_password

# 비밀번호 확인
if not check_password():
    st.stop()



def connect_db():
    """SQLite 데이터베이스 연결"""
    return sqlite3.connect("DB/publisher_evaluation.db")


def get_gdp_weight(country):
    """국가별 GDP 보정 비중 조회"""
    conn = connect_db()
    try:
        query = "SELECT `제곱근 변환` FROM gdp_weights WHERE 국가 = ?"
        result = pd.read_sql_query(query, conn, params=[country])
        if not result.empty:
            return result['제곱근 변환'].iloc[0]
        else:
            return 1.0  # 기본값
    finally:
        conn.close()


def get_currency_weights():
    """환율 정보 조회 """
    conn = connect_db()
    try:
        query = f"SELECT 통화, `USD 대비 상대 가치` FROM annual_currency_weights WHERE 연도 = {YEAR}"
        result = pd.read_sql_query(query, conn)
        return result
    finally:
        conn.close()

# 고정(종합출판사, 시 전문 출판사, 퍼블릭토메인)
def get_ahp_table():
    """AHP 테이블 생성 (데이터만 반환)"""
    ahp_df = pd.DataFrame({
        '연간매출': [0.141],
        '출판종수_연간': [0.085],
        '출판종수_총계': [0.073],
        '선인세': [0.315],
        '인세': [0.145],
        '초판계약부수': [0.240],
    })
    return ahp_df

def calculate_statistics_from_db(category):
    """evaluation_results 테이블에서 기초 통계량 계산"""
    conn = connect_db()
    try:
        # DB에서 필요한 컬럼 데이터 가져오기 (NULL 제외, 지정된 카테고리만)
        query = """
            SELECT 
                연간매출_가중치_적용,
                출판종수_연간,
                출판종수_총계,
                선인세_환율_GDP_보정,
                인세_평균,
                초판부수
            FROM evaluation_results
            WHERE 카테고리 = ?
              AND (연간매출_가중치_적용 IS NOT NULL
               OR 출판종수_연간 IS NOT NULL
               OR 출판종수_총계 IS NOT NULL
               OR 선인세_환율_GDP_보정 IS NOT NULL
               OR 인세_평균 IS NOT NULL
               OR 초판부수 IS NOT NULL)
        """
        df = pd.read_sql_query(query, conn, params=[category])
        
        # 컬럼 매핑: DB 컬럼명 -> summary_stat_df 컬럼명
        column_mapping = {
            '연간매출_가중치_적용': 's연간매출_가중치',
            '출판종수_연간': 's출판종수_연간',
            '출판종수_총계': 's출판종수_총계',
            '선인세_환율_GDP_보정': 's선인세_가중치_USD',
            '인세_평균': 's인세',
            '초판부수': 's초판부수'
        }
        
        base_data = {}
        
        for db_col, stat_col in column_mapping.items():
            # NULL이 아닌 값만 추출
            data = df[db_col].dropna()
            
            if len(data) == 0:
                # 데이터가 없으면 None으로 채움
                base_data[stat_col] = [None] * 10
                continue
            
            # 기초 통계량 계산
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            min_val = data.min()
            max_val = data.max()
            iqr = q3 - q1
            
            # 이상선 계산
            상한_이상선 = q3 + 1.5 * iqr
            하한_이상선 = q1 - 1.5 * iqr
            
            # 윈저화된 값 계산
            # 윈저화된_최소값 = MAX(하한_이상선, MIN(데이터))
            # 윈저화된_최대값 = MIN(상한_이상선, MAX(데이터))
            윈저화된_최소값 = max(하한_이상선, min_val)
            윈저화된_최대값 = min(상한_이상선, max_val)
            
            # 급간 계산: (윈저화된_최대값 - 윈저화된_최소값) / 10
            급간 = (윈저화된_최대값 - 윈저화된_최소값) / 10
            
            base_data[stat_col] = [
                q1, q3, iqr, min_val, max_val,
                상한_이상선, 하한_이상선,
                윈저화된_최소값, 윈저화된_최대값, 급간
            ]
        
        return base_data
    finally:
        conn.close()


def calculate_weight_table(ahp_df, 연간매출_기본값, GDP_가중치, 선인세_환율, 인세_MIN, 인세_MAX, 초판부수, 출판종수_연간, 출판종수_총계, category):
    """출판사 총계 계산 함수"""
    
    # (1) 파생 변수 계산
    derived_dict = {}
    
    # 연간매출_가중치_적용 계산
    if 연간매출_기본값 is not None and GDP_가중치 is not None:
        derived_dict['연간매출_가중치_적용'] = [연간매출_기본값 * GDP_가중치]
    
    # 출판종수_연간
    if 출판종수_연간 is not None:
        derived_dict['출판종수_연간'] = [출판종수_연간]
    
    # 출판종수_총계
    if 출판종수_총계 is not None:
        derived_dict['출판종수_총계'] = [출판종수_총계]
    
    # 선인세_가중치_GDP 계산 (퍼블릭도메인 제외)
    if category != '퍼블릭도메인' and 선인세_환율 is not None and GDP_가중치 is not None:
        derived_dict['선인세_가중치_GDP'] = [선인세_환율 * GDP_가중치]
    
    # 인세 계산 (퍼블릭도메인 제외)
    if category != '퍼블릭도메인' and 인세_MIN is not None and 인세_MAX is not None:
        derived_dict['인세_평균'] = [(인세_MIN + 인세_MAX) / 2]
    
    # 초판부수
    if 초판부수 is not None:
        derived_dict['초판부수'] = [초판부수]
    
    derived_df = pd.DataFrame(derived_dict) if derived_dict else pd.DataFrame()
    
    # (2) 기초 통계량 - DB에서 계산
    base_data = calculate_statistics_from_db(category=category)
    
    index = ["Q1", "Q3", "IQR", "MIN", "MAX", "상한_이상선", "하한_이상선", "윈저화된_최소값", "윈저화된_최대값", "급간"]
    summary_stat_df = pd.DataFrame(base_data, index=index)
    
    print(f"summary_stat_df: {summary_stat_df}")
    
    # (3) 가중치 범위 계산
    weight_bounds_dict = {}
    
    # 연간매출_가중치
    if '연간매출_가중치_적용' in derived_df.columns:
        연간매출_가중치_적용 = derived_df['연간매출_가중치_적용'].iloc[0]
        상한_이상선 = summary_stat_df.loc['상한_이상선', 's연간매출_가중치']
        윈저화된_최소값 = summary_stat_df.loc['윈저화된_최소값', 's연간매출_가중치']
        급간 = summary_stat_df.loc['급간', 's연간매출_가중치']
        
        # None 체크 및 타입 변환
        if (pd.notna(연간매출_가중치_적용) and pd.notna(상한_이상선) and 
            pd.notna(윈저화된_최소값) and pd.notna(급간)):
            상한_이상선 = float(상한_이상선)
            윈저화된_최소값 = float(윈저화된_최소값)
            급간 = float(급간)
            연간매출_가중치_적용 = float(연간매출_가중치_적용)
            
            min_value = np.minimum(연간매출_가중치_적용, 상한_이상선)
            diff = min_value - 윈저화된_최소값
            division = diff / 급간
            floor_value = np.floor(division)
            plus_one = floor_value + 1
            clipped = np.clip(plus_one, 1, 10)
            
            weight_bounds_dict['연간매출_가중치'] = clipped
    
    # 출판종수_연간
    if '출판종수_연간' in derived_df.columns:
        출판종수_연간_값 = derived_df['출판종수_연간'].iloc[0]
        상한_이상선 = summary_stat_df.loc['상한_이상선', 's출판종수_연간']
        윈저화된_최소값 = summary_stat_df.loc['윈저화된_최소값', 's출판종수_연간']
        급간 = summary_stat_df.loc['급간', 's출판종수_연간']
        
        if (pd.notna(출판종수_연간_값) and pd.notna(상한_이상선) and 
            pd.notna(윈저화된_최소값) and pd.notna(급간)):
            상한_이상선 = float(상한_이상선)
            윈저화된_최소값 = float(윈저화된_최소값)
            급간 = float(급간)
            출판종수_연간_값 = float(출판종수_연간_값)
            
            weight_bounds_dict['출판종수_연간'] = np.clip(
                np.floor(
                    (np.minimum(출판종수_연간_값, 상한_이상선) - 
                    윈저화된_최소값) / 급간
                ) + 1,
                1, 10
            )
    
    # 출판종수_총계
    if '출판종수_총계' in derived_df.columns:
        출판종수_총계_값 = derived_df['출판종수_총계'].iloc[0]
        상한_이상선 = summary_stat_df.loc['상한_이상선', 's출판종수_총계']
        윈저화된_최소값 = summary_stat_df.loc['윈저화된_최소값', 's출판종수_총계']
        급간 = summary_stat_df.loc['급간', 's출판종수_총계']
        
        if (pd.notna(출판종수_총계_값) and pd.notna(상한_이상선) and 
            pd.notna(윈저화된_최소값) and pd.notna(급간)):
            상한_이상선 = float(상한_이상선)
            윈저화된_최소값 = float(윈저화된_최소값)
            급간 = float(급간)
            출판종수_총계_값 = float(출판종수_총계_값)
            
            weight_bounds_dict['출판종수_총계'] = np.clip(
                np.floor(
                    (np.minimum(출판종수_총계_값, 상한_이상선) - 
                    윈저화된_최소값) / 급간
                ) + 1,
                1, 10
            )
    
    # 선인세_가중치2
    if '선인세_가중치_GDP' in derived_df.columns:
        선인세_가중치_GDP_값 = derived_df['선인세_가중치_GDP'].iloc[0]
        상한_이상선 = summary_stat_df.loc['상한_이상선', 's선인세_가중치_USD']
        윈저화된_최소값 = summary_stat_df.loc['윈저화된_최소값', 's선인세_가중치_USD']
        급간 = summary_stat_df.loc['급간', 's선인세_가중치_USD']
        
        if (pd.notna(선인세_가중치_GDP_값) and pd.notna(상한_이상선) and 
            pd.notna(윈저화된_최소값) and pd.notna(급간)):
            상한_이상선 = float(상한_이상선)
            윈저화된_최소값 = float(윈저화된_최소값)
            급간 = float(급간)
            선인세_가중치_GDP_값 = float(선인세_가중치_GDP_값)
            
            weight_bounds_dict['선인세_가중치2'] = np.clip(
                np.floor(
                    (np.minimum(선인세_가중치_GDP_값, 상한_이상선) - 
                    윈저화된_최소값) / 급간
                ) + 1,
                1, 10
            )
    
    # 인세_평균
    if '인세_평균' in derived_df.columns:
        인세_평균_값 = derived_df['인세_평균'].iloc[0]
        상한_이상선 = summary_stat_df.loc['상한_이상선', 's인세']
        윈저화된_최소값 = summary_stat_df.loc['윈저화된_최소값', 's인세']
        급간 = summary_stat_df.loc['급간', 's인세']
        
        if (pd.notna(인세_평균_값) and pd.notna(상한_이상선) and 
            pd.notna(윈저화된_최소값) and pd.notna(급간)):
            상한_이상선 = float(상한_이상선)
            윈저화된_최소값 = float(윈저화된_최소값)
            급간 = float(급간)
            인세_평균_값 = float(인세_평균_값)
            
            weight_bounds_dict['인세_평균'] = np.clip(
                np.floor(
                    (np.minimum(인세_평균_값, 상한_이상선) - 
                    윈저화된_최소값) / 급간
                ) + 1,
                1, 10
            )
    
    # 초판부수
    if '초판부수' in derived_df.columns:
        초판부수_값 = derived_df['초판부수'].iloc[0]
        상한_이상선 = summary_stat_df.loc['상한_이상선', 's초판부수']
        윈저화된_최소값 = summary_stat_df.loc['윈저화된_최소값', 's초판부수']
        급간 = summary_stat_df.loc['급간', 's초판부수']
        
        if (pd.notna(초판부수_값) and pd.notna(상한_이상선) and 
            pd.notna(윈저화된_최소값) and pd.notna(급간)):
            상한_이상선 = float(상한_이상선)
            윈저화된_최소값 = float(윈저화된_최소값)
            급간 = float(급간)
            초판부수_값 = float(초판부수_값)
            
            weight_bounds_dict['초판부수'] = np.clip(  
                np.floor(
                    (np.minimum(초판부수_값, 상한_이상선) - 
                    윈저화된_최소값) / 급간
                ) + 1,
                1, 10
            )
    
    weight_bounds_df = pd.DataFrame(weight_bounds_dict, index=[0]) if weight_bounds_dict else pd.DataFrame()
    
    # (4) 최종 결과 계산
    result_dict = {}
    
    if '연간매출_가중치' in weight_bounds_df.columns:
        result_dict['연간매출_가중치'] = weight_bounds_df['연간매출_가중치'] * ahp_df['연간매출'].iloc[0]
    
    if '출판종수_연간' in weight_bounds_df.columns:
        result_dict['출판종수_연간'] = weight_bounds_df['출판종수_연간'] * ahp_df['출판종수_연간'].iloc[0]
    
    if '출판종수_총계' in weight_bounds_df.columns:
        result_dict['출판종수_총계'] = weight_bounds_df['출판종수_총계'] * ahp_df['출판종수_총계'].iloc[0]
    
    if '선인세_가중치2' in weight_bounds_df.columns:
        result_dict['선인세_가중치2'] = weight_bounds_df['선인세_가중치2'] * ahp_df['선인세'].iloc[0]
    
    if '인세_평균' in weight_bounds_df.columns:
        result_dict['인세'] = weight_bounds_df['인세_평균'] * ahp_df['인세'].iloc[0]
    
    if '초판부수' in weight_bounds_df.columns:
        result_dict['초판부수'] = weight_bounds_df['초판부수'] * ahp_df['초판계약부수'].iloc[0]
    
    result_df = pd.DataFrame(result_dict, index=[0]) if result_dict else pd.DataFrame()
    
    # 출판사 총계
    if not result_df.empty:
        total_score = result_df.sum(axis=1).iloc[0]
        # 소수점 세 자리에서 올림
        total_score = np.ceil(total_score * 1000) / 1000
    else:
        total_score = 0.0
    
    return result_df, total_score


def get_category_count(category):
    """카테고리별 총 데이터 개수 조회"""
    conn = connect_db()
    try:
        query = "SELECT COUNT(*) as cnt FROM evaluation_results WHERE 카테고리 = ?"
        result = pd.read_sql_query(query, conn, params=[category])
        return result['cnt'].iloc[0] if not result.empty else 0
    finally:
        conn.close()


def get_grade_ranges_by_percentage(category):
    """점수_총계 기준으로 등급별 비율(30%, 25%, 20%, 15%, 10%)로 나눈 구간 계산"""
    conn = connect_db()
    try:
        # 점수_총계가 NULL이 아닌 데이터만 가져오기 (내림차순)
        query = """
            SELECT 점수_총계
            FROM evaluation_results
            WHERE 카테고리 = ? AND 점수_총계 IS NOT NULL
            ORDER BY 점수_총계 DESC
        """
        df = pd.read_sql_query(query, conn, params=[category])
        
        if df.empty:
            return {
                'rank_ranges': {'S': '', 'A': '', 'B': '', 'C': '', 'D': ''},
                'score_ranges': {'S': '', 'A': '', 'B': '', 'C': '', 'D': ''},
                'counts': {'S': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0}
            }
        
        total_count = len(df)
        
        # 비율에 따른 개수 계산 (반올림)
        s_count = int(round(total_count * 0.30))
        a_count = int(round(total_count * 0.25))
        b_count = int(round(total_count * 0.20))
        c_count = int(round(total_count * 0.15))
        d_count = total_count - s_count - a_count - b_count - c_count  # 나머지
        
        # 순위 범위 계산
        s_start, s_end = 1, s_count
        a_start, a_end = s_count + 1, s_count + a_count
        b_start, b_end = s_count + a_count + 1, s_count + a_count + b_count
        c_start, c_end = s_count + a_count + b_count + 1, s_count + a_count + b_count + c_count
        d_start, d_end = s_count + a_count + b_count + c_count + 1, total_count
        
        # 점수 범위 계산 (해당 구간의 최소/최대 점수)
        s_scores = df.iloc[s_start-1:s_end]['점수_총계']
        a_scores = df.iloc[a_start-1:a_end]['점수_총계'] if a_count > 0 else pd.Series(dtype=float)
        b_scores = df.iloc[b_start-1:b_end]['점수_총계'] if b_count > 0 else pd.Series(dtype=float)
        c_scores = df.iloc[c_start-1:c_end]['점수_총계'] if c_count > 0 else pd.Series(dtype=float)
        d_scores = df.iloc[d_start-1:d_end]['점수_총계'] if d_count > 0 else pd.Series(dtype=float)
        
        # 범위 포맷팅 함수
        def format_rank_range(start, end):
            if start > end:
                return f'{start}위~'
            elif start == end:
                return f'{start}위'
            else:
                return f'{start}~{end}위'
        
        def format_score_range(min_score, max_score):
            if pd.isna(min_score) or pd.isna(max_score):
                return ''
            if min_score == max_score:
                return f'{min_score:.2f} 이상'
            else:
                return f'{min_score:.2f} 이상'
        
        rank_ranges = {
            'S': f'~ {s_end}위' if s_end > 0 else '',
            'A': format_rank_range(a_start, a_end) if a_count > 0 else '',
            'B': format_rank_range(b_start, b_end) if b_count > 0 else '',
            'C': format_rank_range(c_start, c_end) if c_count > 0 else '',
            'D': format_rank_range(d_start, d_end) if d_count > 0 else ''
        }
        
        score_ranges = {
            'S': format_score_range(s_scores.min(), s_scores.max()) if len(s_scores) > 0 else '',
            'A': format_score_range(a_scores.min(), a_scores.max()) if len(a_scores) > 0 else '',
            'B': format_score_range(b_scores.min(), b_scores.max()) if len(b_scores) > 0 else '',
            'C': format_score_range(c_scores.min(), c_scores.max()) if len(c_scores) > 0 else '',
            'D': format_score_range(d_scores.min(), d_scores.max()) if len(d_scores) > 0 else ''
        }
        
        return {
            'rank_ranges': rank_ranges,
            'score_ranges': score_ranges,
            'counts': {'S': s_count, 'A': a_count, 'B': b_count, 'C': c_count, 'D': d_count}
        }
    finally:
        conn.close()


def get_grade_rank_ranges(category):
    """등급별 순위 급간 계산 (점수_총계 기준 비율 분할)"""
    result = get_grade_ranges_by_percentage(category)
    return result['rank_ranges']


def get_grade_and_rank(total_score, category):
    """총점과 순위를 바탕으로 등급 구간표에 맞는 등급과 예상 순위 계산"""
    conn = connect_db()
    try:
        # 먼저 순위 계산 (점수_총계 기준 내림차순)
        rank_query = """
            SELECT COUNT(*) + 1 as 순위
            FROM evaluation_results
            WHERE 카테고리 = ? AND 점수_총계 > ?
        """
        rank_result = pd.read_sql_query(rank_query, conn, params=[category, total_score])
        순위 = int(rank_result['순위'].iloc[0]) if not rank_result.empty else 1
        
        # 등급별 구간 정보 가져오기 (비율 기반)
        grade_ranges = get_grade_ranges_by_percentage(category)
        counts = grade_ranges.get('counts', {})
        
        # 등급별 개수 추출
        s_count = counts.get('S', 0)
        a_count = counts.get('A', 0)
        b_count = counts.get('B', 0)
        c_count = counts.get('C', 0)
        d_count = counts.get('D', 0)
        
        # 순위 범위 계산
        s_start, s_end = 1, s_count
        a_start, a_end = s_count + 1, s_count + a_count
        b_start, b_end = s_count + a_count + 1, s_count + a_count + b_count
        c_start, c_end = s_count + a_count + b_count + 1, s_count + a_count + b_count + c_count
        d_start = s_count + a_count + b_count + c_count + 1
        
        # 순위 기준으로 등급 결정
        if s_end > 0 and 순위 <= s_end:
            등급 = "S"
        elif a_count > 0 and a_start <= 순위 <= a_end:
            등급 = "A"
        elif b_count > 0 and b_start <= 순위 <= b_end:
            등급 = "B"
        elif c_count > 0 and c_start <= 순위 <= c_end:
            등급 = "C"
        elif 순위 >= d_start:
            등급 = "D"
        else:
            # 순위 범위에 해당하지 않으면 D등급
            등급 = "D"
        
        return 등급, f"{순위}위"
    except Exception as e:
        # 예외 발생 시 기본값 반환
        print("예외 발생: ", e)
        return "D", "1위"
    finally:
        conn.close()


# 페이지 설정
st.set_page_config(page_title="출판사 역량분석 시스템", layout="wide")
st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;">
        <h1>출판사 역량분석 시스템</h1>
    </div>
    """, unsafe_allow_html=True)

# CATEGORY 선택 (세션 상태로 관리)
if 'category' not in st.session_state:
    st.session_state.category = '종합출판사'

category_options = ['종합출판사', '시전문출판사', '퍼블릭도메인']
CATEGORY = st.selectbox(
    "**카테고리를 선택하세요:**",
    category_options,
    index=category_options.index(st.session_state.category) if st.session_state.category in category_options else 2,
    key="category_selectbox"
)

# CATEGORY가 변경되면 세션 상태 업데이트 및 페이지 새로고침
if st.session_state.category != CATEGORY:
    st.session_state.category = CATEGORY
    st.rerun()
    
# CSS 스타일링
st.markdown("""
<style>
.main-container {
    padding: 15px;
    margin: 10px;
}
.main-title {
    text-align: left;
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 20px;
}
.section-title {
    text-align: center;
    font-size: 16px;
    font-weight: bold;
    background-color: #e6e6e6;
    padding: 3px;
    margin: 5px 0;
    border: 1px solid #ccc;
}
.red-header {
    background-color: #ffcccc;
    text-align: center;
    font-weight: bold;
    font-size: 11px;
    padding: 2px;
}
.table-container {
    border: 1px solid #000;
    margin: 5px 0;
}
.dataframe {
    font-size: 10px;
}
.notes-box {
    border: 1px solid #000;
    padding: 8px;
    font-size: 12px;
    background-color: #f9f9f9;
    margin-top: 10px;
}
.stDataFrame {
    font-size: 10px;
}
</style>
""", unsafe_allow_html=True)

# 메인 컨테이너
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 메인 타이틀
st.markdown(f'<div class="main-title">({YEAR}) [{CATEGORY}] 등급 계산기</div>', unsafe_allow_html=True)

# 국가 목록 (DB에서 가져오기)
try:
    conn = connect_db()
    country_query = "SELECT DISTINCT 국가 FROM gdp_weights ORDER BY 국가"
    country_df = pd.read_sql_query(country_query, conn)
    country_list = country_df['국가'].tolist()
    conn.close()
except:
    # DB 연결 실패시 기본값
    country_list = ["그리스", "네덜란드", "독일", "러시아", "미국", "방글라데시", "베트남", 
                   "스페인", "아르헨티나", "아제르바이잔", "알바니아", "에티오피아", 
                   "이란", "이스라엘", "이집트", "이탈리아", "인도네시아", "일본", 
                   "중국", "칠레", "튀르키예", "프랑스"]

if CATEGORY == '퍼블릭도메인':
    st.markdown('<p style="color: blue; font-size: 0.9em;">※ 퍼블릭도메인 등급은 인세가 제외되어 계산됩니다.</p>', unsafe_allow_html=True)
else: 
    pass 

# 사용자 입력값들
st.sidebar.header("⌨️입력 값 설정")
COUNTRY = st.sidebar.selectbox("**‧ 국가**", country_list)

# 화폐 선택 옵션
available_currencies = ["USD", "EUR", "KRW", "JPY", "CNY", "GBP", "CHF", "CAD", "AUD"]
단위_연간매출 = st.sidebar.selectbox("**‧ 연간매출 화폐 단위**", available_currencies, index=0)  # USD가 기본값

# 퍼블릭도메인일 때는 선인세, 인세 입력 필드 숨기기
if CATEGORY != '퍼블릭도메인':
    단위_선인세 = st.sidebar.selectbox("**‧ 선인세 화폐 단위**", available_currencies, index=1)  # EUR이 기본값

st.sidebar.markdown("---")
연간매출_기본값 = st.sidebar.number_input(f"**‧ 연간매출 ({단위_연간매출})**", value=35000000, min_value=0)
연간_출판종수 = st.sidebar.number_input("**‧ 연간 출판종수**", value=300, min_value=0)
총_출판종수 = st.sidebar.number_input("**‧ 총 출판종수**", value=1200, min_value=0)

# 퍼블릭도메인일 때는 선인세, 인세 입력 필드 숨기기
if CATEGORY != '퍼블릭도메인':
    선인세_기본값 = st.sidebar.number_input(f"**‧ 선인세 ({단위_선인세})**", value=60000.0, min_value=0.0)
    인세_MIN = st.sidebar.number_input("**‧ 인세 (최소)**", value=0.08, min_value=0.0, max_value=1.0, step=0.01)
    인세_MAX = st.sidebar.number_input("**‧ 인세 (최대)**", value=0.12, min_value=0.0, max_value=1.0, step=0.01)
else:
    선인세_기본값 = None
    인세_MIN = None
    인세_MAX = None
    단위_선인세 = "USD"  # 기본값 설정 (사용되지 않지만 에러 방지)

초판계약부수 = st.sidebar.number_input("**‧ 초판계약부수**", value=10000, min_value=0)

# GDP 보정 비중 가져오기
GDP_보정_비중 = get_gdp_weight(COUNTRY)

# 환율 정보 가져오기
currency_weights = get_currency_weights()

# 선택한 화폐의 USD 대비 상대 가치 찾기 (연간매출 환율 계산용)
try:
    연간매출_currency_rate = currency_weights[currency_weights['통화'] == 단위_연간매출]['USD 대비 상대 가치'].iloc[0]
    연간매출_USD = 연간매출_기본값 * 연간매출_currency_rate  # 선택한 화폐를 USD로 변환
except:
    # 기본값들 (환율 정보가 없을 경우)
    default_rates = {"USD": 1.0, "EUR": 1.093, "KRW": 0.00071, "JPY": 0.0067, 
                    "CNY": 0.138, "GBP": 1.273, "CHF": 1.124, "CAD": 0.721, "AUD": 0.651}
    연간매출_currency_rate = default_rates.get(단위_연간매출, 1.0)
    연간매출_USD = 연간매출_기본값 * 연간매출_currency_rate

# 선택한 화폐의 USD 대비 상대 가치 찾기 (선인세 환율 계산용) - 퍼블릭도메인 제외
if CATEGORY != '퍼블릭도메인':
    try:
        currency_rate = currency_weights[currency_weights['통화'] == 단위_선인세]['USD 대비 상대 가치'].iloc[0]
        선인세_환율 = 선인세_기본값 * currency_rate  # 선택한 화폐를 USD로 변환
    except:
        # 기본값들 (환율 정보가 없을 경우)
        default_rates = {"USD": 1.0, "EUR": 1.093, "KRW": 0.00071, "JPY": 0.0067, 
                        "CNY": 0.138, "GBP": 1.273, "CHF": 1.124, "CAD": 0.721, "AUD": 0.651}
        currency_rate = default_rates.get(단위_선인세, 1.0)
        선인세_환율 = 선인세_기본값 * currency_rate
else:
    선인세_환율 = None
    currency_rate = 1.0  # 기본값 설정 (사용되지 않지만 에러 방지)

# AHP 테이블 가져오기
ahp_df = get_ahp_table()

# 총점 계산
result_df, 총점 = calculate_weight_table(
    ahp_df, 연간매출_USD, GDP_보정_비중, 선인세_환율, 
    인세_MIN, 인세_MAX, 초판계약부수, 연간_출판종수, 총_출판종수, CATEGORY
)

# 등급 및 순위 계산
예상_등급, 예상_순위 = get_grade_and_rank(총점, CATEGORY)

# 첫 번째 행 - 두 개의 섹션
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown(f'<div class="section-title">선택한 국가: {COUNTRY}</div>', unsafe_allow_html=True)
    
    # 투자 계획 테이블 데이터
    invest_data = [
        ['분류 (1)', '분류 (2)', '단위', '값'],
        ['기본정보', '연간매출', f'{단위_연간매출}', f'{연간매출_기본값:,}'],
        ['', '연간 출판종수', '건', f'{연간_출판종수:,}'], 
        ['', '총 출판종수', '건', f'{총_출판종수:,}'],
    ]
    
    # 퍼블릭도메인이 아닐 때만 선인세, 인세 추가
    if CATEGORY != '퍼블릭도메인':
        invest_data.extend([
            ['계약조건', '선인세', f'{단위_선인세}', f'{선인세_기본값:,}'],
            ['', '인세 (최소)', '%', f'{인세_MIN:.2%}'],
            ['', '인세 (최대)', '%', f'{인세_MAX:.2%}'],
        ])
    
    invest_data.append(['계약조건', '초판계약부수', '권', f'{초판계약부수:,}'])
    
    df_invest = pd.DataFrame(invest_data[1:], columns=invest_data[0])
    st.dataframe(df_invest, use_container_width=True, hide_index=True)

with col2:
    # GDP 보정 비중 표시
    st.markdown('<div class="section-title">GDP 보정 비중</div>', unsafe_allow_html=True)
    st.markdown('<div class="red-header">GDP 보정 비중</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center; border: 1px solid #ccc; padding: 5px;">{GDP_보정_비중:.4f}</div>', unsafe_allow_html=True)
    
    # 환율 정보 표시
    st.markdown(f'<div class="section-title">환율 정보 ({YEAR})</div>', unsafe_allow_html=True)
    
    if not currency_weights.empty:
        # 환율 테이블 데이터 준비
        currency_display = currency_weights.copy()
        currency_display['USD 대비 상대 가치'] = currency_display['USD 대비 상대 가치'].apply(lambda x: f"{x:.3f}")
        currency_display.columns = ['기준 통화', '대미 환산율']
        st.dataframe(currency_display, use_container_width=True, hide_index=True)
    else:
        st.write("환율 정보를 불러올 수 없습니다.")

# 두 번째 행
col3, col4 = st.columns([1, 1.5])

with col3:
    st.markdown(f'<div class="section-title">({YEAR}) 등급 분류 결과</div>', unsafe_allow_html=True)
    
    # 예상 순위가 제대로 계산되었는지 확인
    if 예상_순위 is None or 예상_순위 == '':
        예상_순위 = '계산 중...'
    
    target_data = {
        '예상 등급': [예상_등급],
        '총점': [f'{총점:.3f}'],
        '예상 순위': [예상_순위]
    }
    
    df_target = pd.DataFrame(target_data)
    st.dataframe(df_target, use_container_width=True, hide_index=True)

with col4:
    st.markdown('<div class="section-title">(*) 등급 구간표</div>', unsafe_allow_html=True)
    
    # 카테고리별 동적 순위 급간 및 총계 급간 계산
    grade_ranges = get_grade_ranges_by_percentage(CATEGORY)
    rank_ranges = grade_ranges['rank_ranges']
    score_ranges = grade_ranges['score_ranges']
    
    portfolio_data = [
        ['등급 분류', '총계 급간', '순위 급간'],
        ['S', score_ranges['S'] if score_ranges['S'] else '', rank_ranges['S']],
        ['A', score_ranges['A'] if score_ranges['A'] else '', rank_ranges['A']],
        ['B', score_ranges['B'] if score_ranges['B'] else '', rank_ranges['B']],
        ['C', score_ranges['C'] if score_ranges['C'] else '', rank_ranges['C']],
        ['D', score_ranges['D'] if score_ranges['D'] else '', rank_ranges['D']]
    ]
    
    df_portfolio = pd.DataFrame(portfolio_data[1:], columns=portfolio_data[0])
    st.dataframe(df_portfolio, use_container_width=True, hide_index=True)

# 하단 노트 섹션
st.markdown("""
<div class="notes-box">
<strong>(*) 값 입력 방법</strong><br>
- 각 출판사의 번역지원서 내 기재된 항목을 '값' 열에 기입<br>
- '인세'의 경우 최소 값과 최대 값을 각각 기재<br>
  (e.g. 하드커버 8%, 소프트커버 10% => 최소 8% / 최대 10% 기입<br>
- '인세'가 단일 값인 경우, 최소와 최대를 동일하게 기재<br>
  (e.g. 하드커버/소프트커버 10% => 최소 10% / 최대 10% 기입)
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div style="font-size: 0.9em; color: #666;">출판사 역량분석 시스템 v2.0</div>
    <div style="font-size: 0.9em; color: #666;">EC21R&C Inc.</div>
</div>
""", unsafe_allow_html=True)

# 디버깅 정보 (개발시에만 표시)
if st.sidebar.checkbox("디버깅 정보 표시"):
    st.sidebar.write("### 계산 결과")
    st.sidebar.write(f"GDP 보정 비중: {GDP_보정_비중}")
    st.sidebar.write(f"연간매출 화폐: {단위_연간매출}")
    st.sidebar.write(f"연간매출 환율: {연간매출_currency_rate:.6f}")
    st.sidebar.write(f"연간매출 USD 변환: {연간매출_USD:.2f} USD")
    if CATEGORY != '퍼블릭도메인':
        st.sidebar.write(f"선인세 화폐: {단위_선인세}")
        st.sidebar.write(f"선인세 환율: {currency_rate:.6f}")
        st.sidebar.write(f"선인세 USD 변환: {선인세_환율:.2f} USD")
    st.sidebar.write(f"총점: {총점:.3f}")
    if not result_df.empty:
        st.sidebar.write("### 세부 점수")
        st.sidebar.write("아래 테이블을 합산한 값이 총점입니다.")
        st.sidebar.dataframe(result_df.T)