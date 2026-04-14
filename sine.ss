(define two-pi (* 2.0 (acos -1.0)))

(define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))

(define (sigmoid-prime y) (* y (- 1.0 y)))

(define (dot a b)
  (let loop ([a a] [b b] [sum 0.0])
    (if (null? a)
        sum
        (loop (cdr a) (cdr b) (+ sum (* (car a) (car b)))))))

(define (forward-hidden weight-matrix input)
  (map (lambda (row) (sigmoid (dot row input)))
       weight-matrix))

(define (forward-output weight-matrix input)
  (map (lambda (row) (dot row input)) weight-matrix))

(define (update-weights weight-matrix deltas inputs
         learning-rate)
  (map (lambda (row delta)
         (map (lambda (w x) (+ w (* learning-rate delta x)))
              row
              inputs))
       weight-matrix
       deltas))

(define (train-step weights-hidden weights-output inputs
         target learning-rate)
  (let* ([augmented-inputs (append inputs '(1.0))]
         [hidden (forward-hidden weights-hidden augmented-inputs)]
         [augmented-hidden (append hidden '(1.0))]
         [output (car (forward-output
                        weights-output
                        augmented-hidden))]
         [output-delta (- target output)]
         [hidden-deltas (map (lambda (h w)
                               (* (sigmoid-prime h) output-delta w))
                             hidden
                             (list-head
                               (car weights-output)
                               (length hidden)))]
         [updated-output (update-weights
                           weights-output
                           (list output-delta)
                           augmented-hidden
                           learning-rate)]
         [updated-hidden (update-weights
                           weights-hidden
                           hidden-deltas
                           augmented-inputs
                           learning-rate)])
    (values updated-hidden updated-output)))

(define (random-weight) (- (random 1.0) 0.5))

(define (make-weight-matrix rows columns)
  (map (lambda (_)
         (map (lambda (_) (random-weight)) (iota columns)))
       (iota rows)))

(define (make-sine-data point-count)
  (map (lambda (i)
         (let* ([x (/ (* 1.0 i) (- point-count 1))]
                [angle (* x two-pi)])
           (list (list x) (sin angle))))
       (iota point-count)))

(define sine-data (make-sine-data 20))

(define (train-epoch weights-hidden weights-output data
         learning-rate)
  (if (null? data)
      (values weights-hidden weights-output)
      (let-values ([(new-weights-hidden new-weights-output)
                    (train-step weights-hidden weights-output (caar data)
                      (cadar data) learning-rate)])
        (train-epoch
          new-weights-hidden
          new-weights-output
          (cdr data)
          learning-rate))))

(define (train weights-hidden weights-output epochs
         learning-rate)
  (let loop ([weights-hidden weights-hidden]
             [weights-output weights-output]
             [remaining epochs])
    (if (= remaining 0)
        (values weights-hidden weights-output)
        (let-values ([(new-weights-hidden new-weights-output)
                      (train-epoch
                        weights-hidden
                        weights-output
                        sine-data
                        learning-rate)])
          (loop
            new-weights-hidden
            new-weights-output
            (- remaining 1))))))

(define (predict weights-hidden weights-output inputs)
  (let* ([augmented (append inputs '(1.0))]
         [hidden (forward-hidden weights-hidden augmented)]
         [augmented-hidden (append hidden '(1.0))])
    (car (forward-output weights-output augmented-hidden))))

(let* ([weights-hidden (make-weight-matrix 8 2)]
       [weights-output (make-weight-matrix 1 9)])
  (let-values ([(trained-weights-hidden trained-weights-output)
                (train weights-hidden weights-output 50000 0.01)])
    (for-each
      (lambda (sample)
        (let* ([inputs (car sample)]
               [expected (cadr sample)]
               [predicted (predict
                            trained-weights-hidden
                            trained-weights-output
                            inputs)])
          (format #t "x=~f  expected=~f  predicted=~f~%" (car inputs)
            expected predicted)))
      sine-data)))

