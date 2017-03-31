# DBLP_parser.py
import numpy as np
import math
import random
re_sample_acc = open("./dblp.xml", "r+") # input training data
#re_sample_acc = open("./dblp_sort.xml", "r+") # input training data
#re_sample_acc = open("./dblp_test_DY.xml", "r+") # input training data
line = re_sample_acc.readlines()
length = len(line)

file_write_to = open("./output_DBLP.json", "wb")
# print length
# print line[length-1]

# for index in range(0, 10, 1):
# 	print line[index]

# article|inproceedings|proceedings|book|incollection|phdthesis|mastersthesis|www
article_counter = 0
article_end_counter = 0
inproceedings_counter = 0
proceedings_counter = 0
book_counter = 0 
incollection_counter = 0 
phdthesis_counter = 0
mastersthesis_counter = 0 
End_mastersthesis_counter = 0
www_counter = 0
End_www = 0

file_write_to.write("[\n")

def extract_to_list():
	#final_ouput = []
	output_paper= []
	key_counter = 0
	special_case = 0
	article_counter = 0
	for index in range(0, length, 1):
		if index % 1000000 == 0:
			print index
		load_line = line[index]
		if( (" key=") )in load_line:
			# print "Spot an article"
			file_write_to.write("{\n")
			One_paper = [ "0" ] 
			article_counter = article_counter + 1
			element = load_line.split("<")
			if(len(element)==2):
				element_01 = element[1].split(" mdate=")
				One_paper[0] = "type"
			if(len(element)==3):
				element_01 = element[2].split(" mdate=")
				One_paper[0] = "type"
			if(len(element)==4):
				element_01 = element[1].split(" mdate=")
				One_paper[0] = "type"
				element_temp = element[2].split(">")
				special_case = 100
				special_case_key = element_temp[0]
				special_case_value = element_temp[1]
			_type = element_01[0] 
			temp = _type.split(" ")
			_type = temp[0]
			One_paper.append(_type)
			element_02 = element_01[1].split(" key=")
			One_paper.append("mdate")
			element_08 = element_02[0].split("\"")
			_mdate = element_08[1]
			One_paper.append(_mdate)
			element_03 = element_02[1].split(">")
			_key = element_03[0]
			element_07 = element_03[0].split("\"")
			_key = element_07[1]
			One_paper.append("key")
			One_paper.append(_key)
			key_counter = key_counter + 1

			count = 1
			while True:
				one_line = line[index+count].strip()
				if ("</"+_type+">") in one_line:
					break

				# print line[index+count]
				if len(one_line) == 0:
					count += 1
					continue

				try:
					element_04 = one_line.split("</")
					element_05 = element_04[0].split(">")
					element_06 = element_04[1].split(">")
					_value = element_05[1]
					_inner_key = element_06[0]
				except IndexError:
					print one_line
				if(special_case == 100):
					One_paper.append(special_case_key)
					One_paper.append(special_case_value)
					special_case = 0
				One_paper.append(_inner_key)
				One_paper.append(_value)
				count = count + 1
					# output_paper = One_paper

					#final_ouput.append(output_paper)
					#yield output_paper
			paper_dict = to_dict(One_paper)
			yield paper_dict
	#return final_ouput
	#yield output_paper
			# output_paper.append.(One_paper)

def to_dict(alist):
	list_keys = set(['author', 'cite', 'ee', 'editor', 'isbn', 'note', 'url', 'volume',
			 'series', 'cdrom', 'crossref', 'school', 'pages', 'publisher','year'])
	adict = {}
	i = 0
	while i < len(alist):
		key, value = alist[i], alist[i+1]
		i += 2

		if key not in adict:
			if key in list_keys:
				adict[key] = [value]
				continue
			adict[key] = value
			continue
		if key not in list_keys:
			print key
			continue
		adict[key].append(value)
	return adict

# extract_to_list()
for p in extract_to_list():
#print p
	pass
# print extract_to_list()
		
# 		author_counter = 0
# 		editor_counter = 0
# 		for index in range(0, len(One_paper)/2, 1):
# 			if "author" in One_paper[index*2]:
# 				author_counter = author_counter + 1 
# 			if "editor" in One_paper[index*2]:
# 				editor_counter = editor_counter + 1
		
# 		# print "author_counter=", author_counter

# 		if(author_counter < 2  and  editor_counter < 2 ):
# 			for index in range(0, len(One_paper)/2-1, 1):
# 				# print One_paper[index*2], One_paper[index*2+1]
# 				file_write_to.write("'"+One_paper[index*2]+"':'"+One_paper[index*2+1]+"',\n")
# 			file_write_to.write("'"+One_paper[(len(One_paper)/2-1)*2]+"':'"+One_paper[(len(One_paper)/2-1)*2+1]+"'\n")
				
# 		elif(author_counter > 1 and editor_counter > 1):
# 			index = 0
# 			while(index < len(One_paper)/2-1):
# 				# print One_paper[index*2], One_paper[index*2+1]
# 				if(One_paper[index*2] == "author"):
# 					file_write_to.write("'"+One_paper[index*2]+"':[")
# 					for index3 in range(0, author_counter-1, 1):
# 						file_write_to.write("'"+One_paper[index*2+1 + index3*2]+"',")
# 					file_write_to.write("'"+One_paper[index*2+1 + author_counter-1*2]+"'")
# 					file_write_to.write("],\n")
# 					index = index + author_counter-1
# 				if(One_paper[index*2] == "editor"):
# 					file_write_to.write("'"+One_paper[index*2]+"':[")
# 					for index3 in range(0, editor_counter-1, 1):
# 						file_write_to.write("'"+One_paper[index*2+1 + index3*2]+"',")
# 					file_write_to.write("'"+One_paper[index*2+1 + editor_counter-1*2]+"'")
# 					file_write_to.write("],\n")
# 					index = index + editor_counter-1
# 				else:
# 					file_write_to.write("'"+One_paper[index*2]+"':'"+One_paper[index*2+1]+"',\n")
# 				index = index + 1
# 			file_write_to.write("'"+One_paper[(len(One_paper)/2-1)*2]+"':'"+One_paper[(len(One_paper)/2-1)*2+1]+"'\n")
# 		elif(author_counter > 1 and editor_counter <2  ):
# 			index = 0
# 			while(index < len(One_paper)/2-1):
# 				# print One_paper[index*2], One_paper[index*2+1]
# 				if(One_paper[index*2] == "author"):
# 					file_write_to.write("'"+One_paper[index*2]+"':[")
# 					for index3 in range(0, author_counter-1, 1):
# 						file_write_to.write("'"+One_paper[index*2+1 + index3*2]+"',")
# 					file_write_to.write("'"+One_paper[index*2+1 + author_counter-1*2]+"'")
# 					file_write_to.write("],\n")
# 					index = index + author_counter-1
# 				else:
# 					file_write_to.write("'"+One_paper[index*2]+"':'"+One_paper[index*2+1]+"',\n")
# 				index = index + 1
# 			file_write_to.write("'"+One_paper[(len(One_paper)/2-1)*2]+"':'"+One_paper[(len(One_paper)/2-1)*2+1]+"'\n")
# 		elif(author_counter < 2 and editor_counter >1 ):
# 			index = 0
# 			while(index < len(One_paper)/2-1):
# 				# print One_paper[index*2], One_paper[index*2+1]
# 				if(One_paper[index*2] == "editor"):
# 					file_write_to.write("'"+One_paper[index*2]+"':[")
# 					for index3 in range(0, editor_counter-1, 1):
# 						file_write_to.write("'"+One_paper[index*2+1 + index3*2]+"',")
# 					file_write_to.write("'"+One_paper[index*2+1 + editor_counter-1*2]+"'")
# 					file_write_to.write("],\n")
# 					index = index + editor_counter-1
# 				else:
# 					file_write_to.write("'"+One_paper[index*2]+"':'"+One_paper[index*2+1]+"',\n")
# 				index = index + 1
# 			file_write_to.write("'"+One_paper[(len(One_paper)/2-1)*2]+"':'"+One_paper[(len(One_paper)/2-1)*2+1]+"'\n")
			

				

# 		file_write_to.write("},\n")


# file_write_to.write("{} ]")

# print "Code Ended successfully", "key_counter=", key_counter




		# print line[index+count+1]
		# print "End of an article"



		# while "</article>" not in load_line:
		# 	index_inner = index_inner +1
		# 	print "index_inner", index_inner
# 	elif "</article>" in load_line:
# 		article_end_counter = article_end_counter + 1
# 	elif "<inproceedings mdate=" in load_line:
# 		inproceedings_counter = inproceedings_counter + 1  
# 	elif "<proceedings mdate=" in load_line:
# 		proceedings_counter = proceedings_counter + 1  
# 	elif "<book mdate=" in load_line:
# 		book_counter = book_counter + 1 
# 	elif "<incollection mdate=" in load_line:
# 		incollection_counter = incollection_counter + 1 
# 	elif "<phdthesis mdate=" in load_line:
# 		phdthesis_counter = phdthesis_counter + 1 
# 	elif "<mastersthesis mdate=" in load_line:
# 		index2 = 0
# 		mastersthesis_counter = mastersthesis_counter + 1
# 		# while "</mastersthesis>" not in load_line:
# 		# 	print line[index+index2]
# 		# 	index2 = index2 + 1
# 		# count = 0
# 		# while (count < 9):
# 		# 	print 'The count is:', count
# 		# 	count = count + 1
# 		# print "Good bye!"



# 	elif "</mastersthesis>" in load_line:
# 		End_mastersthesis_counter = End_mastersthesis_counter + 1
# 		print load_line
# 	elif "<www mdate=" in load_line:
# 		www_counter = www_counter + 1
# 	elif "</www>" in load_line:
# 		End_www = End_www + 1 


# print article_counter, "article", article_end_counter, "</article>", article_counter-article_end_counter 
# print inproceedings_counter, "inproceedings_counter"
# print proceedings_counter, "proceedings"
# print book_counter, "book_counter"
# print incollection_counter, "incollection_counter"
# print phdthesis_counter, "phdthesis_counter"
# print mastersthesis_counter, "mastersthesis_counter", End_mastersthesis_counter
# print www_counter, "www", End_www
