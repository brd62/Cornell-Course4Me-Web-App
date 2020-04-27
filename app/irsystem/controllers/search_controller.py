from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *
import pickle

project_name = "Course4me"
net_id = "Brady Dickens (brd62), Micah Wallingford (mjw286), Sam Rosenthal (ser259), Julian Londono (jl3358)"

classes_dict = pickle.load(open( "./class_roster_api_dict.pickle", "rb"))
classes_list = set()
for class_key in classes_dict.keys():
	for class_name in classes_dict[class_key]["subject-number"]:
		classes_list.add(class_name.upper())


@irsystem.route('/', methods=['GET'])
def search():
	keyword_query = request.args.get('keyword_search')
	# professor_query = request.args.get('professor_search')
	class_query = request.args.get('class_search')
	if keyword_query:
		data = getKeywordResults(keyword_query)
		suggestions = getSuggestions(keyword_query)
		if len(data) > 0 :
			output_message = "Results for \"" + keyword_query + "\""
		else:
			output_message = "No results found for \"" + keyword_query + "\""

	elif class_query:
		print(class_query)
		data = getClassResults(class_query)
		suggestions = []
		print(len(data))
		if len(data) > 0 :
			output_message = "Results for "+ class_query
		else:
			output_message = "No results found for \"" + keyword_query + "\""
	else:
		data = []
		suggestions = []
		output_message = ''

	return render_template('search.html', name=project_name, netid=net_id,
							output_message=output_message, data=data, suggestions= suggestions,
							classes_list=classes_list)
