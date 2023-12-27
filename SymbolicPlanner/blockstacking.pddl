(define (domain blockstacking)
    (:predicates ;todo: define predicates here
        (clear ?a); no object on a
        (on ?a ?b); a on b
        (onTable ?a); a is on some obj
    )

    (:action move_x_to_y
        :parameters (?x ?y)
        :precondition (and 
            (clear ?x) (clear ?y) (onTable ?x))
        :effect (and 
            (not (clear ?y)) (on ?x ?y) (not (onTable ?x)))
    )    
)