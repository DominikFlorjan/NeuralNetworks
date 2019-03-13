//Array for training
let training = new Array(2000);
let perceptron;

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
}

//Line
function f(x){
    let y = 0.3 * x + 0.4;
    return y 
}