(ns scicloj.ml.deep-diamond.boston
  (:require
     [clojure.java.io :as io]
     [clojure.data.csv :as csv]
     [clojure.string :as string]
     [uncomplicate.commons [core :refer [with-release let-release info view]]]
     [uncomplicate.neanderthal
      [core :refer [transfer transfer! view-vctr native view-ge
                    cols mv! rk! raw col row nrm2 scal! ncols dim rows]]
      [real :refer [entry! entry]]
      [native :refer [native-float fv]]
      [random :refer [rand-uniform!]]
      [math :as math :refer [sqrt]]]
     [uncomplicate.diamond
      [tensor :refer [*diamond-factory* tensor connector transformer
                      desc revert shape input output view-tz batcher]]
      [dnn :refer [sum activation inner-product fully-connected network init! train! cost infer!]]]
   
     [uncomplicate.diamond.internal.neanderthal.factory :refer [neanderthal-factory]]))
   


  
(def boston-housing-raw
  (csv/read-csv (slurp (io/resource "boston-housing-prices/boston-housing.csv"))))

(def boston-housing
  (doall (shuffle (map #(mapv (fn [^String x] (Double/valueOf x)) %) (drop 1 boston-housing-raw)))))


(def x-train (map (partial take 13) (take 404 boston-housing)))
(def y-train (map (partial drop 13) (take 404 boston-housing)))
(def x-test (map (partial take 13) (drop 404 boston-housing)))
(def y-test (map (partial drop 13) (drop 404 boston-housing)))


(defn standardize!
  ([a!]
   (let-release [row-means (raw (col a! 0))]
                (when (< 0 (dim a!))
                  (with-release [ones (entry! (raw (row a! 0)) 1)]
                    (mv! (/ -1.0 (ncols a!)) a! ones row-means)
                    (standardize! row-means a!)))
                row-means))
  ([row-means a!]
   (when (< 0 (dim a!))
     (with-release [ones (entry! (raw (row a! 0)) 1)]
       (rk! row-means ones a!)
       (doseq [x-mean (rows a!)]
         (let [s (double (nrm2 x-mean))]
           (if (= 0.0 s)
             x-mean
             (scal! (/ (sqrt (ncols a!)) s) x-mean))))))
   a!))


  
(defn r [layers x-train y-train x-test y-test]
  (let [
        fact (neanderthal-factory)
        x-train-m (transfer native-float x-train)


        y-train-m (transfer native-float y-train)

        x-test-m (transfer native-float x-test)

        y-test-m (transfer native-float y-test)]

    (standardize! x-train-m)
    (standardize! x-test-m)

    (println :x-train-m--dim  (dim x-train-m))
    (println :y-train-m--dim  (dim y-train-m))
    (println :x-test-m--dim  (dim x-test-m))
    (println :y-test-m--dim  (dim y-test-m))



    (with-release [x-tz (tensor fact [404 13] :float :nc)
                   x-mb-tz (tensor fact [16 13] :float :nc)
                   y-tz (tensor fact [404 1] :float :nc)
                   y-mb-tz (tensor fact [16 1] :float :nc)
                   net-bp (network fact x-mb-tz layers)

                   net (init! (net-bp x-mb-tz :adam))
                   net-infer (net-bp x-mb-tz)
                   quad-cost (cost net y-mb-tz :quadratic)
                   mean-abs-cost (cost net-infer y-mb-tz :mean-absolute)
                   x-batcher (batcher x-tz (input net))
                   y-batcher (batcher y-tz y-mb-tz)]

    
      (transfer! x-train-m (view-vctr x-tz))

      (transfer! y-train-m (view-vctr y-tz))
      (train! net x-batcher y-batcher quad-cost 80 [])

      (transfer! net net-infer)
      (net-infer)
      (println :mac (mean-abs-cost)))))

       ;; (println :net (transfer! net))



(def layers-1
  [(fully-connected [64] :relu)
   (fully-connected [64] :relu)
   (fully-connected [1] :linear)])


(r
 layers-1
 x-train y-train x-test y-test)
;; => {:mean-absolute-cost 2.2119651890205594E33}



(def layers-2
  [;; (fully-connected [64] :relu)
   ;; (fully-connected [64] :relu)
   (fully-connected [1] :linear)])


(r
 layers-2
 x-train y-train x-test y-test)
;; => {:mean-absolute-cost 2.2119634868530054E33}


