** Developer: Abhishek Kumar 
** Date: August 28th 2026

** What We Built: Multi-Layer Perceptron (MLP)
		Our model uses fully connected layers (nn.Linear) and an activation function (nn.ReLU). This architecture is technically called a Multi-Layer Perceptron (MLP).
			• How it works: An MLP treats the input features as a single, fixed group of numbers. It processes them all at the same time, calculating feature interactions in parallel. You can think of it like looking at a coordinate (x,y,zx,y,z) and running the numbers through a calculator to get a single output.
        Best for: Structured, fixed-size data, like images (processed after feature extraction), tabular data, or simple numerical predictions where the order of the inputs doesn't fundamentally change the meaning.

** The Code **

1. src/data_loader.py (The Organizer)
	• What it is: This file is the team's Librarian and Sifter.
	• What it does: When we get a massive pile of raw toys (our data), the librarian doesn't just dump it on the floor. It has a job: 
		1. Load It: It takes the big pile of data and gets it organized into neat boxes.
		2. Clean It: It checks every toy and organizes which ones are "Features" (the clues, like the shape or color) and which one is the "Target" (the answer we are trying to find, like whether the toy belongs in the Train bin or the Toy bin).
		3. Split It: Finally, it carefully divides the organized toys into two piles: a "Practice Pile" (for training) and a "Test Pile" (for checking if the robot actually learned it).
🧠 2. src/model.py (The Brain)
	• What it is: This file is the robot's Brainchip.
	• What it does: It's the actual structure that learns. We designed it using layers, think of it like a set of filters: 
		1. Input Filter (FC1): The first filter takes the input clues (your features) and processes them, looking for simple patterns.
		2. Hidden Filter (ReLU): This is like the robot pausing and magically simplifying the patterns it found.
		3. Output Filter (FC2): The last filter takes those simple patterns and spits out a final answer (the prediction).
	• Goal: The code only builds the brain; it doesn't teach it yet!
🧑‍🏫 3. src/trainer.py (The Coach)
	• What it is: This file is the expert Coach or Teacher.
	• What it does: The model's brain is powerful, but it needs direction! The Coach runs the practice: 
		1. Guess: In one round, the Coach gives the Model the Practice Pile and says, "Make a guess!" (This is the Forward Pass).
		2. Check: The Coach then compares the Model's guess to the correct answer (the "truth"). This comparison gives us a "Mistake Score" (the Loss).
		3. Adjust: The Coach tells the Model: "You were wrong here, try to fix this!" (This is Backpropagation).
		4. Learn: The Model tweaks its internal settings (its weights) slightly to be a little bit better next time. This whole round is one Epoch.
🗺️ 4. run_training.py (The Conductor)
	• What it is: This file is the Conductor of the Orchestra.
	• What it does: It makes sure everything happens in the right order! It doesn't do any of the hard work itself; it just tells the other three modules what to do and when. 
		1. Set Up: It calls the Librarian to get the data ready.
		2. Assemble: It asks the Model's Brainchip and the Coach's Teaching tools to pair up.
		3. Run: It runs the Coach through the practice loop (the epochs) many times.
Grade: Finally, it tells the Brainchip to take a final, calm test on the Test Pile to give us the final grade (the Test Loss)!

