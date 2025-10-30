import pandas as pd, ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 데이터 로드
places = pd.read_csv("장소최종데이터2.csv")
users = pd.read_csv("더미_사용자_일정데이터_with_userid.csv")

# 2. category 문자열 처리
def parse_category(x):
    try:
        v = ast.literal_eval(str(x))
        return " ".join(v) if isinstance(v, list) else str(v)
    except Exception:
        return str(x)

places["feature_text"] = (
    places["place_type"].fillna("") + " " + places["category"].apply(parse_category)
)
users["feature_text"] = users["theme"].astype(str) + " " + users["relation"].astype(str)

# 3. TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X_place = vectorizer.fit_transform(places["feature_text"])
X_user = vectorizer.transform(users["feature_text"])

# 4. 사용자별 추천
recs = []
for i, row in users.iterrows():
    sims = cosine_similarity(X_user[i], X_place).ravel()
    top_idx = sims.argsort()[-5:][::-1]
    top_places = places.loc[top_idx, ["place", "place_type", "address"]].copy()
    top_places.insert(0, "user_id", row["user_id"])
    top_places.insert(1, "name", row["name"])
    top_places["relation"] = row["relation"]
    top_places["theme"] = row["theme"]
    top_places["similarity_score"] = sims[top_idx]
    recs.append(top_places)

recommendations = pd.concat(recs, ignore_index=True)
recommendations.to_csv("콘텐츠기반_추천결과_테마_구성원_속성_수정.csv", index=False, encoding="utf-8-sig")
