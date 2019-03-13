//Array for training
let training = new Array(2000);
let perceptron;

var count = 0;

//Setting up coordinate space
let xmin = -1;
let xmax = 1;
let ymin = -1;
let ymax = 1;

function setup(){
    createCanvas(600, 600);
    //3 weights because of 3 inputs (x, y, bias)
    perceptron = new Perceptron(3, 0.01);

    //Random set of training points
    for(let i = 0; i < training.length; i++){
        let x = random(xmin, xmax);
        let y = random(ymin, ymax);
        let answer = 1; 
        if(y < f(x)) answer = -1; 
        training[i] ={
            input: [x, y, 1],
            output: answer
        };
    }

}

function draw(){
    background(25);
    
    strokeWeight(1);
    stroke(255);
    let x1 = map(xmin, xmin, xmax, 0, width);
    let y1 = map(f(xmin), ymin, ymax, height, 0);
    let x2 = map(xmax, xmin, xmax, 0, width);
    let y2 = map(f(xmax), ymin, ymax, height, 0);
    line(x1,y1,x2,y2);

    strokeWeight(2);
    let weights = perceptron.getWeights();
    x1=xmin;
    y1 = (-weights[2] - weights[0]*x1)/weights[1];
    x2 = xmax; 
    y2 = (-weights[2] - weights[0] * x2) / weights[1];

    x1 = map(x1, xmin, xmax, 0, width);
    y1 = map(y1, ymin, ymax, height, 0);
    x2 = map(x2, xmin, xmax, 0, width);
    y2 = map(y2, ymin, ymax, height, 0);
    line(x1,y1,x2,y2);

    perceptron.train(training[count].input, training[count].output);
    count = (count + 1) % training.length;

    for(let i = 0; i < count; i++){
        stroke(255);
        strokeWeight(1);
        fill(255);
        let guess = perceptron.feedForward(training[i].input);
        if(guess > 0) noFill();

        let x = map(training[i].input[0], xmin, xmax, 0, width);
        let y = map(training[i].input[1], ymin, ymax, height, 0);
        ellipse(x, y, 8, 8);
    }
    
}

//Line
function f(x){
    let y = 1 * x + 0.5;
    return y 
}