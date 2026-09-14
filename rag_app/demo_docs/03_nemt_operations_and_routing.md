# NEMT Operations and Route Optimization

Non-emergency medical transportation (NEMT) providers drive patients to scheduled appointments such as dialysis, chemotherapy, physical therapy and routine doctor visits. Unlike an ambulance service, NEMT trips are booked in advance and are not for emergencies.

Every trip has a level of service that decides which vehicle can carry the rider. Ambulatory riders can walk and ride in a standard sedan or van. Wheelchair riders need a vehicle with a lift or ramp and wheelchair securement. Stretcher riders must lie flat and need a stretcher van with trained staff.

A trip usually has two legs: the outbound leg takes the rider from home to the appointment, and the return leg brings them back. When the appointment end time is not known in advance, the return is booked as a will-call, and the rider phones the dispatcher when they are ready to be picked up.

Dispatchers track on-time performance and no-shows. A no-show is recorded when the rider is not at the pickup location within the allowed wait time, which is commonly around 10 to 15 minutes depending on the broker contract.

Assigning trips to drivers is a vehicle routing problem with pickups and deliveries. Each trip must be picked up before it is dropped off, by the same vehicle. The pickup must happen inside a time window so the rider reaches the appointment on time, and the vehicle must never carry more riders or wheelchairs than it has seats and wheelchair positions.

Route optimization tools such as Google OR-Tools search for assignments that respect these hard constraints while minimizing soft costs such as total driving time, dead-head miles driven empty, and late arrivals. Finding the perfect answer is NP-hard, so solvers use heuristics and stop at a time limit with the best solution found.

The solver needs a travel-time matrix that gives the driving time between every pair of stops. Routing engines such as OSRM compute these matrices from OpenStreetMap road data. Straight-line distance underestimates real driving time, especially around rivers, highways and one-way streets.

Trips that cannot be assigned without breaking a hard constraint are returned as unassigned, together with a reason such as no vehicle with a wheelchair position being free inside the pickup window. Dispatchers then fix the data, relax a policy, or add a vehicle and run the solver again.
