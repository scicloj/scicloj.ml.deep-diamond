(ns scicloj.ml.deep-diamond.boston-housing-data
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.string :as string]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.tensor :as dst]

   [tech.v3.tensor :as tens]))


(def mb-size 512)
(defn read-imdb-master
  ([]
   (->> (io/resource "imdb-sentiment/imdb_master.csv")
        (slurp)
        (csv/read-csv)
        (drop 1)
        (filter (fn  [[_ _ _ assesment]] (not  (= "unsup" assesment))))
        shuffle))

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



(defn encode-review [word-map review x y]
  (let [[_ words sentiment] (split-review review)]

    (doseq [idx (map word-map words)]
      (when idx (tens/mset! x idx 1.0)))
    (tens/mset! y 0 (case sentiment "neg" 0 "pos" 1)))
  x)







(defn encode-reviews [wmap reviews max-vocab]
  (let [in (tens/native-tensor [(count reviews) max-vocab])
        out (tens/native-tensor [(count reviews) 1])]
    (doall (map #(encode-review wmap %1 %2 %3) reviews (tens/rows in) (tens/rows out)))
    [in out]))




(defn boston-data [max-vocab data-size train-size]
  (let [wvec (word-vec (pmap split-review (read-imdb-master)) max-vocab)
        wmap (word-map wvec)
        review-split (split-at train-size (read-imdb-master data-size))

        train-reviews-text (-> review-split first shuffle)
        test-reviews-text (-> review-split second shuffle)

        train-reviews (encode-reviews wmap train-reviews-text max-vocab)
        test-reviews   (encode-reviews wmap test-reviews-text max-vocab)
        ds-train
        (->
         (first train-reviews)
         dst/tensor->dataset
         (ds/add-column (ds/new-column :y (second train-reviews)))
         (ds-mod/set-inference-target :y))
        ds-test (->
                 (first test-reviews)
                 dst/tensor->dataset
                 (ds/add-column (ds/new-column :y (second test-reviews)))
                 (ds-mod/set-inference-target :y))]
    {:train-ds ds-train
     :test-ds ds-test}))


(comment
  (-> train-reviews-text count)

  (def a-review
    (-> train-reviews-text first))



  (def train-in (tens/native-tensor [ (count train-reviews-text) max-vocab]))
  (def train-out (tens/native-tensor [1 (count train-reviews-text)]))

  (encode-review wordmap a-review)
  (-> train-in tens/columns count))






(defn decode-review [word-vec code-vec]
  (filter identity
          (map #(if (< 0.5 (tens/mget code-vec %))
                  (word-vec %)
                  nil)
               (range (count code-vec)))))
