(define pi (acos -1.0))

(define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))

(define (sigmoid-prime y) (* y (- 1.0 y)))

(define (dot a b)
  (let loop ([a a] [b b] [sum 0.0])
    (if (null? a)
        sum
        (loop (cdr a) (cdr b) (+ sum (* (car a) (car b)))))))

(define (softmax logits)
  (let* ([max-logit (apply max logits)]
         [exps (map (lambda (z) (exp (- z max-logit))) logits)]
         [sum-exps (apply + exps)])
    (map (lambda (e) (/ e sum-exps)) exps)))

(define (forward-hidden weight-matrix input)
  (map (lambda (row) (sigmoid (dot row input)))
       weight-matrix))

(define (forward-output weight-matrix input)
  (softmax
    (map (lambda (row) (dot row input)) weight-matrix)))

(define (update-weights weight-matrix deltas inputs
         learning-rate)
  (map (lambda (row delta)
         (map (lambda (w x) (+ w (* learning-rate delta x)))
              row
              inputs))
       weight-matrix
       deltas))

(define (train-step weights-hidden weights-output inputs
         target-class learning-rate)
  (let* ([augmented-inputs (append inputs '(1.0))]
         [hidden (forward-hidden weights-hidden augmented-inputs)]
         [augmented-hidden (append hidden '(1.0))]
         [probabilities (forward-output
                          weights-output
                          augmented-hidden)]
         [one-hot (map (lambda (k) (if (= k target-class) 1.0 0.0))
                       (iota (length weights-output)))]
         [output-deltas (map - one-hot probabilities)]
         [hidden-deltas (map (lambda (h j)
                               (* (sigmoid-prime h)
                                  (apply
                                    +
                                    (map (lambda (delta wo-row)
                                           (* delta (list-ref wo-row j)))
                                         output-deltas
                                         weights-output))))
                             hidden
                             (iota (length hidden)))]
         [updated-output (update-weights
                           weights-output
                           output-deltas
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

(define (make-spiral-data points-per-class)
  (apply
    append
    (map (lambda (class-index)
           (map (lambda (i)
                  (let* ([t (/ (* 1.0 i) points-per-class)]
                         [angle (+ (* t 4.0 pi)
                                   (* class-index (/ (* 2.0 pi) 3.0)))]
                         [x (* t (cos angle))]
                         [y (* t (sin angle))])
                    (list (list x y) class-index)))
                (iota points-per-class)))
         (iota 3))))

(define spiral-data (make-spiral-data 50))

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
                        spiral-data
                        learning-rate)])
          (loop
            new-weights-hidden
            new-weights-output
            (- remaining 1))))))

(define (argmax lst)
  (let loop ([rest (cdr lst)]
             [max-val (car lst)]
             [max-index 0]
             [index 1])
    (if (null? rest)
        max-index
        (if (> (car rest) max-val)
            (loop (cdr rest) (car rest) index (+ index 1))
            (loop (cdr rest) max-val max-index (+ index 1))))))

(define (predict weights-hidden weights-output inputs)
  (let* ([augmented (append inputs '(1.0))]
         [hidden (forward-hidden weights-hidden augmented)]
         [augmented-hidden (append hidden '(1.0))])
    (argmax (forward-output weights-output augmented-hidden))))

(define (accuracy weights-hidden weights-output data)
  (let ([correct (length
                   (filter
                     (lambda (sample)
                       (= (predict
                            weights-hidden
                            weights-output
                            (car sample))
                          (cadr sample)))
                     data))])
    (/ (* 100.0 correct) (length data))))

(let* ([weights-hidden (make-weight-matrix 16 3)]
       [weights-output (make-weight-matrix 3 17)])
  (let-values ([(trained-weights-hidden trained-weights-output)
                (train weights-hidden weights-output 10000 0.1)])
    (format
      #t
      "accuracy: ~f%~%"
      (accuracy
        trained-weights-hidden
        trained-weights-output
        spiral-data))))

