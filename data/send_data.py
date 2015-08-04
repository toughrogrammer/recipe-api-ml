
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