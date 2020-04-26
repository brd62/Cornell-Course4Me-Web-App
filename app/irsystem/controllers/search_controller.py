from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *

project_name = "Course4me"
net_id = "Brady Dicken (brd62), Micah Wallingford (mjw286), Sam Rosenthal (ser259), Julian Londono (jl3358)"

@irsystem.route('/', methods=['GET'])
def search():
	keyword_query = request.args.get('keyword_search')
	professor_query = request.args.get('professor_search')

	if not keyword_query:
		data = []
		suggestions = []
		output_message = ''
	else:
		data = getResults(keyword_query)
		suggestions = getSuggestions(keyword_query)
		print(suggestions)
		if len(data) > 0 :
			output_message = "Results for \"" + keyword_query + "\""
		else:
			output_message = "No results found for \"" + keyword_query + "\""
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, suggestions= suggestions)



