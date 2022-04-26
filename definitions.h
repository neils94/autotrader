#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <curl/curl.h>
#include <pybind11/pybind11.h>


using namespace std; //make standard library accessible to use names
namespace py = pybind11;


class Agent {

Agent::Agent() //agent class constructor

//parameters to change in Agent class include action size(int)
//in order to get an output size for neural network 
//no need to include state size because state size is continuous
//and neural networks will acept the output of CNN instead of
//traditional state size with linear networks input
private:
typedef int action_size; //private access to action_size
typedef float state; //private access to the vectorized state
typedef int batch_size; //batch size for updates to networks
typedef int buffer_size; //buffer size for replay buffer



public:




}

template <typename i> 



template 

class Environment{

    Environment::Environment() //class constructor for environment
        
        //bearer token string to be used in API calls
        private:
        const str bearer_token("c00f3da14e736b95bce848501bfdea7a-54f731eeda7ba20287dae157a23943ba");

        
        
        
    // CREATE CLASS FOR INTERACTIONS WITH ENVIRONMENT:
        
        
        // FUNCTION TO CUE TAKING TRADES (Oanda -> Docs -> Order)
        
            // Parameters for opening Order (/v3/accounts/{accountID}/orders) inlclude:
        
                //Autorization: bearer token (string)
                //Accept-DateTime format:  Date Time formatted in Unix (string)
                //Account ID (string)
    
        
    curl -H "Authorization: Bearer bearer_token https://api-fxtrade.oanda.com/v3/accounts/{accountID}/positions"  
        
        // FUNCTION THAT RETURNS ALL POSITIONS AND CURRENT BALANCE (v3/accounts/{accountID}/positions):
        
        // Parse out most recent position
        
            //Parameters for Position (Oanda -> Docs -> Position) (v3/accounts/{accountID}/positions) function include:
        
                //Authorization bearer token (string)
                //path: AccountID (string)
          
    curl -H "Authorization: Bearer bearer_token https://api-fxtrade.oanda.com/v3/accounts/{accountID}"
        
        
            //Parameters for Account balance (Oanda -> Docs -> Account) (v3/accounts/{accountID}):
                
                //Authorization bearer token (string) 
                //Accept-DateTime format:  Date Time formatted in Unix (string)
                ///path: AccountID (string)



// REWARD FUNCTION:
int reward(float trade_result){
        
    if (done = True){
        
        curl -H "Authorization: Bearer bearer_token https://api-fxtrade.oanda.com/v3/accounts/{accountID}"
        
        return math::log(trade_result);
  
        } 




}
