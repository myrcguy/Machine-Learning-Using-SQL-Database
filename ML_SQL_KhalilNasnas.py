import pymysql
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from matplotlib import pyplot as plt
import graphviz
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def Decision_Tree_Model_C_P_S(df, calories, protein, sugars):
    
	feature_columns = ['calories','protein','sugars']
	X = df[feature_columns]
	y = df['name'].apply(str)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

	clf = DecisionTreeClassifier()
	clf = clf.fit(X.values, y)
	y_pred = clf.predict([[calories,protein,sugars]])
	print("Predicted cereal:",y_pred[0])

def Decision_Tree_Model_F_C_R(df, fiber, carb, rating):
    
	feature_columns = ['fiber','carbo','rating']
	X = df[feature_columns]
	y = df['name'].apply(str)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

	clf = DecisionTreeClassifier()
	clf = clf.fit(X.values, y)
	y_pred = clf.predict([[fiber,carb,rating]])
	print("Predicted cereal:",y_pred[0])

def Decision_Tree_Model_G_B_A(df, Glucose, BMI, Age):
    
	feature_columns = ['Glucose','BMI','Age']
	X = df[feature_columns]
	y = df['Outcome']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

	clf = DecisionTreeClassifier()
	clf = clf.fit(X.values, y)
	y_pred = clf.predict([[Glucose,BMI,Age]])
	if(y_pred[0] == 0):
		print("Person tested negative for diabetes.")
	elif(y_pred[0] == 1):
		print("Person tested positive for diabetes.")

def main():
	db = pymysql.connect(host='localhost', user='mp', passwd = 'eecs118', db='flights')
	cur = db.cursor()
	df = pd.read_csv('./cereal.csv')
	dfDiabetes = pd.read_csv('./diabetes.csv')
	while True:
		print("Welcome to the Query data Base!")
		print("1. Query: Find the cheapest non-stop flight given airports and a date.  ")
		print("2. Query: Find the flight and seat information for a customer. ")
		print("3. Query: Find all non-stop flights for an airline. ")
		print("4. Query: Find flights that are able to fly on weekdays and cost less than or equal to user input.")
		print("5. Query: Find the phone numbers of customers that use a specific airline.")
		print("6. Query: Classify a cereal based on calories, protein, and sugar.")
		print("7. Query: Find a predicted rating value given a cereal column and a value to be predicted.")
		print("8. Query: Classify a cereal based on fiber, carbo, and rating.")
		print("9: Query: Classify if a person has diabetes based on Glucose, Bmi, and Age.")
		print("10: Query: Predict BMI values based on given glucose level.")
		print("11: Enter q to exit program.")
		queryValue = input("Enter a number (1-10) to select which query you would like to choose, or q for exit: ")
		if(queryValue == "1"):
				QueryOne(cur)
		if(queryValue == "2"):
				QueryTwo(cur)
		if(queryValue == "3"):
				QueryThree(cur)
		if(queryValue == "4"):
				QueryFour(cur)
		if(queryValue == "5"):
				QueryFive(cur)
		if(queryValue == "6"):
				QuerySix(df)
		if(queryValue == "7"):
				QuerySeven(df)
		if(queryValue == "8"):
				QueryEight(df)
		if(queryValue == "9"):
				QueryNine(dfDiabetes)
		if(queryValue == "10"):
				QueryTen(dfDiabetes)
		if(queryValue == "Q" or queryValue == "q"):
				break
	db.close()
	

#('G4155',1,'2018-01-28',3,28,'SCK','535PM','IWA','819PM'), example
def QueryOne(curValue):
	departureCode = input("Please enter the airport code for the departure airport: ")
	destinationCode = input("Please enter the airport code for the destination airport: ")
	dateOfFlight = input("What is the date of the flight in yyyy-mm-dd?: ")
	#departureCode = "'" + departureCode + "'"
	#destinationCode = "'" + destinationCode + "'"
	#dateOfFlight = "'" + dateOfFlight + "'"
	#sql = """SELECT Flight_number, Amount FROM Leg_instance NATURAL JOIN Fare   
	#		where leg_date = '2018-08-05' AND Departure_airport_code= 'SFO' AND Arrival_airport_code = 'ORD' 
	#		ORDER BY Amount"""
	
	sql = """SELECT Flight_number, Amount FROM Leg_instance NATURAL JOIN Fare   
			where leg_date = %s AND Departure_airport_code= %s  AND Arrival_airport_code = %s 
			ORDER BY Amount"""

	#sql = "SELECT * FROM Leg_instance where leg_date = " + dateOfFlight + "and Departure_airport_code = " + departureCode + "and Arrival_airport_code = " + destinationCode
	#sql = "SELECT * FROM Leg_instance where leg_date = " + dateOfFlight 
	#sql = "SELECT * FROM Airplane_type where Company = 'Boeing'"
	curValue.execute(sql, (dateOfFlight,departureCode, destinationCode))
	#for row in curValue.fetchall():
		#listOfFlights.append(row)
		#print(row)
	AirplaneData = curValue.fetchone()
	#print(AirplaneData[0])
	#print(AirplaneData[1])
	print("The cheapest flight is",AirplaneData[0],"and the cost is $", AirplaneData[1],".")


	#print(listOfFlights)
	return

def QueryTwo(curValue):
	customerName = input("Please enter the customer's name: ")
	sql = """SELECT Flight_number, Seat_number FROM Seat_reservation
			where Customer_name = %s """
	flights = []
	curValue.execute(sql, customerName)
	for row in curValue.fetchall():
		print("The flight number is", row[0], "and the seat number is",row[1],".") 
	print()
	return

def QueryThree(curValue):
	airlineName = input("What is the name of the airline? ")
	sql = """SELECT Flight_number FROM Flight_leg NATURAL JOIN Flight
			where leg_number = 1 and Airline = %s """
	
	flightNumbers = []
	curValue.execute(sql,airlineName)
	for row in curValue.fetchall():
		flightNumbers.append(row[0])
	
	print("The non-stop flights are: ")
	for flights in flightNumbers:
		print(flights + ",")
	return

def QueryFour(curValue):
	#Find the flights that are able to fly on weekdays and cost less than user input
	maxCost = input("What is the most amount you can spend on a flight: ")
	sql = """SELECT Flight_number, Amount from FLight NATURAL JOIN Fare
			where Weekdays = 'yes' AND Amount <= %s """
	curValue.execute(sql, maxCost)
	for row in curValue.fetchall():
		print("Flight number",row[0],"Price $",row[1])
	return

def QueryFive(curValue):
	#Find the phone numbers of customers that use a specific airline
	airline = input("What airline do you want to see customer phone numbers from: ")
	sql = """SELECT Customer_name, Customer_phone from Seat_reservation NATURAL JOIN Flight
			where Airline = %s"""
	curValue.execute(sql,airline)
	for row in curValue.fetchall():
		print("Customer name:", row[0], "Customer Phone Number:",row[1])
	return

def QuerySix(df):
	#Classify the cereal based on calories, protein, and sugar values  
	calories = input('Enter cereal calorie value: ')
	protein = input("Enter cereal protein value: ")
	sugars = input("Enter cereal sugar value: ")
	#print(user_list)
	Decision_Tree_Model_C_P_S(df, calories, protein, sugars)


	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
	#clf = DecisionTreeClassifier()
	#clf = clf.fit(X_train, y_train)
	#y_pred = clf.predict(X_test)



def QuerySeven(df):
	#Give a ceral column and the rating column and a value that is to be predicted
	input_X =  input("Enter the first column of Cereal as your X axis: ")
	input_predictor = input("Enter a value based on the Column to have a predicted rating: ")
	input_y = df["rating"]
	X_train = None
	Y_train = None
	X_test = None
	Y_test = None
	X = df[input_X]
	Y = df["rating"]
	X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.1)
  
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	model = linear_model.LinearRegression().fit(X_train.reshape(-1,1),Y_train)

	X_test = np.array(X_test)
	X_test[0] = input_predictor
	print(X_test[0])
	Y_pred = model.predict(X_test.reshape(-1,1))
	print("Rating based on prediction: ",Y_pred[0])


def QueryEight(df):
	#Classify the cereal based on fiber, carbo, and rating  
	fiber = input('Enter cereal fiber value: ')
	carb = input("Enter cereal carb value: ")
	rating = input("Enter cereal rating value: ")
	#print(user_list)
	Decision_Tree_Model_F_C_R(df, fiber, carb, rating)

def QueryNine(df):
	#classify if a person has diabetes based on Glucose, Bmi, and Age
	Glucose = input("Enter the person's glucose value: ")
	BMI = input("Enter the persons BMI value: ")
	Age = input("Enter the age of the person: ")

	Decision_Tree_Model_G_B_A(df,Glucose,BMI,Age)

def QueryTen(df):
	#Build a linear regression to predict BMI values based on Glucose level
	input_predictor = input("Enter a BMI value to predict an estimate of the Glucose level: ")

	X = df["BMI"]
	Y = df["Glucose"]
	X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.1)
  
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	model = linear_model.LinearRegression().fit(X_train.reshape(-1,1),Y_train)
	X_test = np.array(X_test)
	X_test[0] = input_predictor
	print(X_test[0])
	Y_pred = model.predict(X_test.reshape(-1,1))
	print("Glucose level predicted: ",Y_pred[0])
if __name__ == "__main__":

	main()
