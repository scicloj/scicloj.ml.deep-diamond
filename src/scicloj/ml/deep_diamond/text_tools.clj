(ns scicloj.ml.deep-diamond.text-tools
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.string :as string]
   [tech.v3.dataset.neanderthal :as neanderthal]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.tensor :as dst]

   [tech.v3.tensor :as tens]))


(def max-vocab 10000)
(def data-size 50000)
(def train-size 25000)
(def mb-size 512)
(def test-size (- data-size train-size))
(defn read-imdb-master
  ([]
   (->> (io/resource "imdb-sentiment/imdb_master.csv")
        (slurp)
        (csv/read-csv)
        (drop 1)))
  ([cnt]
   (take cnt (read-imdb-master))))

(defn word-frequencies [reviews]
  (apply merge-with + (pmap #(frequencies (% 1)) reviews)))

(defn word-vec [reviews cnt]
  (->> (word-frequencies reviews)
       (sort-by val >)
       (map #(% 0))
       (take cnt)
       (into [])))

(defn word-map [word-vector]
  (into {} (map #(vector (word-vector %) %) (range (count word-vector)))))

(defn split-review [review]
  (vector (review 1) (string/split (review 2) #" ") (review 3)))

(def wvec (word-vec (pmap split-review (read-imdb-master)) max-vocab))
(def wmap (word-map wvec))



(def review-split (split-at train-size (read-imdb-master data-size)))



(defn encode-review [word-map review x y]
  (let [[_ words sentiment] (split-review review)]
    (doseq [idx (map word-map words)]
      (when idx (tens/mset! x idx 1.0)))
    (tens/mset! y 0 (case sentiment "neg" 0 "pos" 1)))
  x)


(defn encode-reviews [wmap reviews]
  (let [in (tens/native-tensor [max-vocab (count reviews)])
        out (tens/native-tensor [1 (count reviews)])]
    (doall (map #(encode-review wmap %1 %2 %3) reviews (tens/columns in) (tens/columns out)))
    [in out]))



(def train-reviews (encode-reviews wmap (-> review-split first shuffle)))
(def test-reviews   (encode-reviews wmap (-> review-split second shuffle)))

(def ds-train
  (->
   (first train-reviews)
   dst/tensor->dataset
   (ds/add-column (ds/new-column :y (second train-reviews)))
   (ds-mod/set-inference-target :y)))


(def ds-test
  (->
   (first test-reviews)
   dst/tensor->dataset
   (ds/add-column (ds/new-column :y (second test-reviews)))
   (ds-mod/set-inference-target :y)))





(comment
  (defn decode-review [word-vec code-vec]
    (filter identity
            (map #(if (< 0.5 (tens/mget code-vec %))
                    (word-vec %)
                    nil)
                 (range (count code-vec))))))






(println :ds-train (ds/shape ds-train))
(println :ds-test (ds/shape ds-test))
