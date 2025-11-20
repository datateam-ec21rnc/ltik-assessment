"""
인증 관련 공통 모듈
"""
import streamlit as st
import sys
import os

# 현재 파일의 디렉토리를 sys.path에 추가하여 import 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def check_password():
    """비밀번호 확인 함수"""
    # 세션 상태 초기화
    if 'password_correct' not in st.session_state:
        st.session_state.password_correct = False
    
    # 비밀번호가 맞으면 True 반환
    if st.session_state.password_correct:
        return True
    
    # 비밀번호 입력 폼
    st.subheader("인증")
    password_input = st.text_input(':closed_lock_with_key: **비밀번호를 입력하세요:**', type="password", key="password_input")
    
    if st.button("로그인"):
        # secrets.toml에서 비밀번호 가져오기
        try:
            correct_password = st.secrets["password"]
            if password_input == correct_password:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ 비밀번호가 올바르지 않습니다.")
        except KeyError:
            st.error("❌ 비밀번호 설정이 없습니다. .streamlit/secrets.toml 파일을 확인하세요.")
    
    return False

