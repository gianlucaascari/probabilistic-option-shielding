action(0)::action(left).
action(1)::action(right).
action(2)::action(up).
action(3)::action(down).
action(4)::action(stop).

% Other-agent sensors (indices 0-3)
sensor_value(0)::other_agent(left).
sensor_value(1)::other_agent(right).
sensor_value(2)::other_agent(up).
sensor_value(3)::other_agent(down).

% Military zone sensors (indices 4-7)
sensor_value(4)::military_area(left).
sensor_value(5)::military_area(right).
sensor_value(6)::military_area(up).
sensor_value(7)::military_area(down).

transition(left,left).
transition(right,right).
transition(up,up).
transition(down,down).

% Unsafe if the next cell is occupied by another agent or is a military zone
unsafe_next :- action(A), transition(A,NextPos), other_agent(NextPos).
unsafe_next :- action(A), transition(A,NextPos), military_area(NextPos).

safe_next :- action(stop).
safe_next :- \+ unsafe_next.
