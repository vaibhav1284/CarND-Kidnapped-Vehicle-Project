#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  // Set the number of particles
  num_particles = 500;

  // Normal (Gaussian) distribution for x, y, and theta
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Create num_particles amount of particles
  for (int i = 0; i < num_particles; ++i) {
	  
	// Create temp particle
    Particle temp_particle;
    
    // Setting values to particles
    temp_particle.id = i;
    temp_particle.x = dist_x(gen);
    temp_particle.y = dist_y(gen);
    temp_particle.theta = dist_theta(gen);
    temp_particle.weight = 1.0;
    particles.push_back(temp_particle);
    weights.push_back(1);
  }
  
  // Flag is initialized
  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  
  // Normal (Gaussian) distribution for x, y, and theta
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  // Iterating through all particles
  for (int i = 0; i < num_particles; ++i)
  {
	// Checking for yaw rates near zero values  
    if (fabs(yaw_rate) < 0.001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    
    // Add gaussian noise 
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  } 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  
  float min_dist;
  float min_index;
  float obs_x;
  float obs_y;
  float distance;
  
  // Iterating through observations
  for(int i = 0; i < observations.size(); ++i)
  {
    obs_x = observations[i].x;
    obs_y = observations[i].y;
    min_dist = numeric_limits<float>::max();
    
    // Iterating through all predictions
    for(int j = 0; j < predicted.size(); ++j)
    {
      distance = dist(obs_x, obs_y, predicted[j].x, predicted[j].y);
      if (distance < min_dist)
      {
        min_index = j;
        min_dist = distance;
      }
    }

    // Index of smallest distance
    observations[i].id = min_index;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) 
{
  
  float obs_x,obs_y,map_x,map_y;
  vector<LandmarkObs> map_observations(observations.size());
  vector<LandmarkObs> map_landmarks_in_range;

  // Iterating through particles
  for (int i = 0; i < num_particles; ++i)
  { 
	//Translaing observations to map coordinates
    for (int j = 0; j < observations.size(); ++j)
    {
      obs_x = observations[j].x;
      obs_y = observations[j].y;
      map_x = particles[i].x + obs_x*cos(particles[i].theta) - obs_y*sin(particles[i].theta);
      map_y = particles[i].y + obs_x*sin(particles[i].theta) + obs_y*cos(particles[i].theta);
      map_observations[j].x = map_x;
      map_observations[j].y = map_y;
    }
	
	// Resetting landmark ranges
	if (map_landmarks_in_range.size() > 0)
      map_landmarks_in_range.clear();
    
	//Setting Landmarks to relevant values
    for (int k = 0; k < map_landmarks.landmark_list.size(); ++k)
    {
      float landmark_id = map_landmarks.landmark_list[k].id_i;
      float landmark_x = static_cast<double>(map_landmarks.landmark_list[k].x_f);
      float landmark_y = static_cast<double>(map_landmarks.landmark_list[k].y_f);
	  
      if (dist(particles[i].x, particles[i].y, landmark_x, landmark_y) <= sensor_range)
      {
        LandmarkObs landmark = {landmark_id, landmark_x, landmark_y};
        map_landmarks_in_range.push_back(landmark);
      }
    }
    
    dataAssociation(map_landmarks_in_range, map_observations);
    particles[i].weight = 1;
   
    for (int p = 0; p < map_observations.size(); ++p)
    {

      // Store observation coordinates
      float x = map_observations[p].x;
      float y = map_observations[p].y;
      
      // Obtain index of associated landmark (nearest neighbor) and get landmark coordinates
      int index = map_observations[p].id;
      
	  float prob = exp(-((pow(x - map_landmarks_in_range[index].x, 2))/(2* (pow(std_landmark[0], 2)) ) + (pow(y - map_landmarks_in_range[index].y, 2))/
						(2* (pow(std_landmark[1], 2))))) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
						
      particles[i].weight *= prob; 
    }
	
  }
}

void ParticleFilter::resample() {
  
  default_random_engine gen;
  vector<Particle> resample(num_particles);
  
  //Clear weights and reinitialize
  weights.clear();
  for (int i = 0; i < num_particles; ++i)
  {
    weights.push_back(particles[i].weight);
  }
  
  // Discrete distribution
  discrete_distribution<size_t> dist_gen(weights.begin(), weights.end());
  
  // Iterating on the number of particles
  for (int i = 0; i < num_particles; ++i)
  {
	  resample[i] = particles[dist_gen(gen)];
  }
  particles = resample;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}