"""
Send sample query to prediction engine
"""

import predictionio
engine_client = predictionio.EngineClient(url="http://localhost:8000")

print "Sending query..."

# for i in range(1, 23):
#   print "User: " + str(i)
#   print engine_client.send_query(
#     {
#       "user": "u" + str(i), 
#       "num": 20
#       #"blackList" : ["i43"]
#     }
#   )
#   
print engine_client.send_query(
    {
      "user": "u21" ,
      "num": 100
      #"blackList" : ["i43"]
    }
  )

# print engine_client.send_query(
#   {
#     "user": "u11",
#     "num": 10,
#     "categories": ["c4", "c3"],
#     "whiteList": ["i1", "i23", "i26", "i31"],
#     "blackList": ["i21", "i25", "i30"]
#   }
# )
