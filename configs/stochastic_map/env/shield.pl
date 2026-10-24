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

transition(left,left).
transition(right,right).
transition(up,up).
transition(down,down).

% Unsafe if the next cell is occupied by another agent
unsafe_next :- action(A), transition(A,NextPos), other_agent(NextPos).

safe_next :- action(stop).
safe_next :- \+ unsafe_next.
