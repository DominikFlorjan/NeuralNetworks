// Perceptron is basicly just one node with some inputs (linear classifier)

class Perceptron{
    constructor(numberOfWeights, learningRate){
        this.weights = new Array(numberOfWeights);

        //Ofc initialization weights with random number
        for(let i=0; i < this.weights.length; i++){
            this.weights[i] = random(-1,1);
        }
        this.lr = learningRate;
    }

    train(inputs, targets){
        let guess = this.feedForward(inputs);
        //classic error calculation, error = desired value - what perceptron guessed in feedforward
        let error = targets - guess;
        //Adjust the weights
        for(let i = 0; i < this.weights.length; i++){
            this.weights[i] += this.lr * error * inputs[i];
        }
    }

    feedForward(inputs){
        let sum = 0;
        for(let i= 0; i < this.weights.length; i++){
            sum+= this.weights[i]*inputs[i]; 
        } 
        return this.activationFunction(sum);
    }

    //Very basic activation function
    activationFunction(x){
        if(x>0) return 1;
        else return -1;
    }

    getWeights(){
        return this.weights;
    }
}