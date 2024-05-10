(ns scicloj.ml.deep-diamond.core)

(set! *unchecked-math* :warn-on-boxed)

(use '[uncomplicate.neanderthal core native])


(def x (dv 1 2 3))
(def y (dv 10 20 30))
(println
 (dot x y))

