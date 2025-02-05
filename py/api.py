import requests

# API 요청 URL 및 파라미터
url = "https://apis.data.go.kr/B551011/KorPetTourService/areaBasedList"
params = {
    "serviceKey": "y2sUtnpqTn56%2BpKD1ZYVNNfS8w%2FJUu4ojflDHkRXtyGUVAEOjvFOSlaDwBkduBc4pmBiHzMSxbY0DzCzN3NSmg%3D%3D",  # URL 인코딩된 인증키
    "numOfRows": 10,          # 한 페이지에 반환할 결과 수
    "pageNo": 1,              # 페이지 번호
    "MobileOS": "ETC",        # OS 구분
    "MobileApp": "AppTest",   # 앱 이름
    "arrange": "A",           # 제목순 정렬
    "listYN": "Y",            # 목록 반환 여부
    "contentTypeId": 12,      # 관광 타입 (12 = 관광지)
    "areaCode": 34,           # 지역 코드 (충청남도)
    "sigunguCode": 16,        # 시군구 코드 (계룡시)
    "cat1": "A01",            # 대분류 코드 (자연관광)
    "cat2": "A0101",          # 중분류 코드 (자연생태관광지)
    "cat3": "A01011800",      # 소분류 코드 (저수지)
    "_type": "json"           # 응답 형식 (JSON)
}

# 요청 보내기
try:
    response = requests.get(url, params=params)
    response.raise_for_status()  # HTTP 오류 발생 시 예외 처리
    data = response.json()       # JSON 응답 파싱

    # 데이터 출력
    print(data)
except requests.exceptions.RequestException as e:
    print(f"API 요청 실패: {e}")



# 예상 결과
# {
#   "response": {
#     "header": {
#       "resultCode": "0000",
#       "resultMsg": "OK"
#     },
#     "body": {
#       "items": {
#         "item": [
#           {
#             "addr1": "충청남도 계룡시 두마면 입암길 218",
#             "addr2": "",
#             "areacode": "34",
#             "cat1": "A01",
#             "cat2": "A0101",
#             "cat3": "A01011800",
#             "contentid": "2654766",
#             "contenttypeid": "12",
#             "createdtime": "20200429010534",
#             "firstimage": "http://tong.visitkorea.or.kr/cms/resource/12/2675812_image2_1.jpg",
#             "firstimage2": "http://tong.visitkorea.or.kr/cms/resource/12/2675812_image2_1.jpg",
#             "cpyrhtDivCd": "Type3",
#             "mapx": "127.2555380499",
#             "mapy": "36.2433505799",
#             "mlevel": "6",
#             "modifiedtime": "20241206104928",
#             "sigungucode": "16",
#             "tel": "",
#             "title": "계룡 입암저수지",
#             "zipcode": "32842"
#           }
#         ]
#       },
#       "numOfRows": 1,
#       "pageNo": 1,
#       "totalCount": 1
#     }
#   }
# }