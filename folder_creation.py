import os

# 애플리케이션 폴더 자동 생성기
def makeFolders(project_names):
    os.makedirs(project_names, exist_ok=True)
    os.makedirs(os.path.join(project_names, 'server_codes'), exist_ok=True)
    os.makedirs(os.path.join(project_names, 'model_codes'), exist_ok=True)
    os.makedirs(os.path.join(project_names, 'ui_codes'), exist_ok=True)


if __name__ == '__main__':

    # 다음과 같이 사용하자.
    makeFolders('자연어 프레임워크')