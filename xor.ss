(define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))

(define (sigmoid-prime y) (* y (- 1.0 y)))

(define (dot a b)
  (let loop ([a a] [b b] [sum 0.0])
    (if (null? a)
        sum
        (loop (cdr a) (cdr b) (+ sum (* (car a) (car b)))))))

(define (forward weight-matrix input)
  (map (lambda (row) (sigmoid (dot row input)))
       weight-matrix))

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
         [hidden (forward weights-hidden augmented-inputs)]
         [augmented-hidden (append hidden '(1.0))]
         [output (car (forward weights-output augmented-hidden))]
         [output-delta (* (- target output) (sigmoid-prime output))]
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

(define xor-data
  '(((0.0 0.0) 0.0)
     ((0.0 1.0) 1.0)
     ((1.0 0.0) 1.0)
     ((1.0 1.0) 0.0)))

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
                        xor-data
                        learning-rate)])
          (loop
            new-weights-hidden
            new-weights-output
            (- remaining 1))))))

(define (predict weights-hidden weights-output inputs)
  (let* ([augmented (append inputs '(1.0))]
         [hidden (forward weights-hidden augmented)]
         [augmented-hidden (append hidden '(1.0))])
    (car (forward weights-output augmented-hidden))))

(let* ([weights-hidden (make-weight-matrix 2 3)]
       [weights-output (make-weight-matrix 1 3)])
  (let-values ([(trained-weights-hidden trained-weights-output)
                (train weights-hidden weights-output 10000 0.5)])
    (for-each
      (lambda (sample)
        (let* ([inputs (car sample)]
               [expected (cadr sample)]
               [predicted (predict
                            trained-weights-hidden
                            trained-weights-output
                            inputs)])
          (format #t "~a xor ~a -> ~f (expected ~a)~%"
            (inexact->exact (car inputs)) (inexact->exact (cadr inputs))
            predicted (inexact->exact expected))))
      xor-data)))

