from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *

project_name = "Course4me"
net_id = "Brady Dicken (brd62), Micah Wallingford (mjw286), Sam Rosenthal (ser259), Julian Londono (jl3358)"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Results for \"" + query + "\""
		data = test(query)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



