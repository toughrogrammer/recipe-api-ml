# 레시피 추천 알고리즘

## 설명

PredictionIO의 'E-Commerce Recommendation Engine Template'과 'Similar Product Engine Template'을 결합, 수정, 보완하여 완성한 추천 알고리즘 엔진입니다.

이 엔진은 다음과 같은 기능들을 제공합니다.

* 기존 유저의 최근 행동을 분석하여 비슷한(선호할만한) 아이템을 추천
* 기존 유저에게 비슷한 아이템이 없을 경우 대중적으로 인기있는 아이템을 추천
* 새로운 유저에게는 대중적으로 인기있는 아이템을 추천
* 보지 않은 아이템만 추천할 수 있음 (optional)
* 카테고리, 화이트리스트, 블랙리스트 필터링 가능 (optional)
* 아이템을 일시 접근불가 설정 가능 (optional)

## 사용법

### Event Data Requirements

* Users' view events
* Users' like events
* Users' cancel_like events
* Items' with categories properties
* Constraint unavailable set events

### Input Query

* UserID
* Num of items to be recommended
* List of white-listed item categories (optional)
* List of white-listed itemIDs (optional)
* List of black-listed itemIDs (optional)

### Output PredictedResult

* A ranked list of recommended itemIDs

## Sending data & query example

send_data.py:

```

import predictionio
import argparse
import random

SEED = 3

def send_event(client):
  random.seed(SEED)
  count = 0
  print client.get_status()
  print "Sending data..."

  client.create_event(
    event="$set",
    entity_type="user",
    entity_id="u21"
  )

  client.create_event(
    event="$set",
    entity_type="item",
    entity_id="i101",
    properties={
      "categories" : "c3"
    }
  )

  client.create_event(
    event="$set",
    entity_type="constraint",
    entity_id="unavailableItems",
    properties={
      "items" : ["i4", "i14", "i11"]
    }
  )

  client.create_event(
    event="view",
    entity_type="user",
    entity_id="u7",
    target_entity_type="item",
    target_entity_id="i80"
  )

  client.create_event(
    event="like",
    entity_type="user",
    entity_id="u7",
    target_entity_type="item",
    target_entity_id="i80"
  )

  client.create_event(
    event="cancel_like",
    entity_type="user",
    entity_id="u8",
    target_entity_type="item",
    target_entity_id="i80"
  )

  print "Complete"


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Import sample data for e-commerce recommendation engine")
  parser.add_argument('--access_key', default='SGjdQqdg3dKfoLtRrtePkSe2yQzCXcpuqSdwGbdHSTj0770tfHNhQ0NV5KCJ5nu3')
  parser.add_argument('--url', default="http://localhost:7070")

  args = parser.parse_args()
  print args

  client = predictionio.EventClient(
    access_key=args.access_key,
    url=args.url,
    threads=5,
    qsize=500)
  send_event(client)

```


send-query.py:

```

import predictionio
engine_client = predictionio.EngineClient(url="http://localhost:8000")

print "Sending query..."

print engine_client.send_query(
  {
    "user": "u10", 
    "num": 50
  }
)

print engine_client.send_query(
  {
    "user": "u11",
    "num": 10,
    "categories": ["c4", "c3"],
    "whiteList": ["i1", "i23", "i26", "i31"],
    "blackList": ["i21", "i25", "i30"]
  }
)

```

