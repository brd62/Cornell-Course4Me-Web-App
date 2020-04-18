def test(query):
  data = []
  for s in range(len(query)):
    data.append(query[len(query)-1-s])
  return data;