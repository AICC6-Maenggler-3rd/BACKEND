import math

# 하버사인 함수
def haversine_distance(lat1, lon1, lat2, lon2):
  """
  두 지점 간의 거리를 킬로미터 단위로 계산 (하버사인 공식)
  Args:
    lat1: 출발지 위도
    lon1: 출발지 경도
    lat2: 도착지 위도
    lon2: 도착지 경도
  Returns:
    두 지점 간의 거리 (km)
  """
  R = 6371  # 지구의 반지름 (km)
  
  # 위도와 경도를 라디안으로 변환
  lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
  
  # 위도와 경도의 차이
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  
  # 하버사인 공식
  a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
  c = 2 * math.asin(math.sqrt(a))
  
  return R * c