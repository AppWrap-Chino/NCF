Absolutely! Here's a basic task using TensorFlow Playground that's tailored for beginners, along with a simple grading rubric that rewards efficiency:

**Task:**

1. **Dataset:** Use the "Circle" dataset in TensorFlow Playground. It's a simple classification task where you need to separate blue and orange points.

2. **Goal:**  Achieve a test loss that's close to (but not necessarily lower than) the training loss. This means your model is learning well but isn't overfitting (memorizing the training data too much).

3. **Key:** Do this with as few neurons in the hidden layers as possible. The fewer neurons you need, the more efficient your model is.

**Here's how students would approach it:**

1. **Start Simple:** Begin with a single hidden layer and just a couple of neurons (like 2 or 3).
2. **Train:** Click "Run" and see how the model performs. Pay attention to how the blue and orange areas change over time.
3. **Adjust:** If it's not doing well (the colored areas don't separate the points), try adding one more neuron to the hidden layer.
4. **Repeat:** Keep adding neurons one at a time until the model separates the points fairly well.  The goal is to get both the training loss and test loss to low values.

**Grading Rubric:**

* **Base Grade:** Start with a base grade (e.g., 80%).
* **Neuron Deductions:** For each neuron used in the hidden layers, deduct a small amount from the base grade (e.g., -2 points per neuron).
* **Performance Bonus:** If the student achieves a test loss that's very close to the training loss, give them a bonus (e.g., +5 points).
* **Example:**  A student uses 3 neurons and achieves a test loss of 0.05 and a training loss of 0.04. Their grade might be: 80 (base) - 6 (neurons) + 5 (bonus) = 79

**Why This Works:**

* **Beginner-Friendly:**  Focuses on the core concept of adjusting model complexity without diving into complex architectures.
* **Efficiency:** Encourages students to find the simplest solution, which is a valuable skill in machine learning.
* **Easy Grading:** The grading rubric is clear and objective, making it easy to assess.

**Important Considerations:**

* **Time Limit:** Set a reasonable time limit for the exam so students don't get stuck trying to perfect their models.
* **Instructions:** Make the instructions very clear and provide an example of what a "good" solution might look like.
* **Open Discussion:** After the exam, discuss different student solutions and why some were more efficient than others.

Let me know if you'd like any modifications to this task or the grading rubric! 
