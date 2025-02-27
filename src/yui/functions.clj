(defn random-response []
  (str (rand-nth (:responses config))))

(defn affirm []
  (str (rand-nth (:affirm config))))

(defn unspace [str]
  (s/replace str #" " "%20"))

(defn escape-underscores [str]
  (s/replace str #"_" "\\_"))

;; is mod?
(defn mod? [user-id]
  (= user-id "885027267401646121"))

(defmacro when-mod [user-id & body]
  (list 'if (mod? user-id)
        (cons 'do body)))

(def model-idx (atom [(jt/plus (jt/local-date-time)
                         (jt/days 1))
                0]))

(def image-model-list ["meta-llama/llama-3.2-90b-vision-instruct:free"
                       "meta-llama/llama-3.2-11b-vision-instruct:free"])

(def model-list [
                 ;; Enter strings with names of OpenRouter models you'd like to use
                 ])


(def ai-prompt
  ;; Enter the prompt for the bot
  )

(def messages-ctx (atom [{:role "system" :content ai-prompt}]))

(defn flush-memory []
  (reset! messages-ctx
          [{:role "system" :content ai-prompt}]))

(defn rotate-model
  ([]
   (swap! model-idx (fn [mdidx]
                      [(jt/plus (jt/local-date-time)
                                (jt/days 1))
                       (mod (inc (first (rest mdidx))) 5)])))
  ([num]
   (reset! model-idx [(jt/plus (jt/local-date-time)
                               (jt/days 1))
                      (mod num 5)])))

(def search-ai-username-list
  (atom (:usernames (edn/read-string (slurp "~/.yui/ai-usernames.edn")))))

(def search-ai-username-assoc
  (atom (:usernames-assoc (edn/read-string (slurp "~/.yui/ai-usernames.edn")))))

(defn search-ai-add-username [username id]
  (let [username-file (edn/read-string (slurp "~/.yui/ai-usernames.edn"))]
    (spit "~/.yui/ai-usernames.edn"
          (binding [*print-length* -1] (prn-str (assoc-in username-file 
                                                          [(keyword "usernames")] 
                                                          (concat ((keyword "usernames")
                                                                   username-file)
                                                                  [username])))))
    (spit "~/.yui/ai-usernames.edn"
          (binding [*print-length* -1] (prn-str (assoc-in username-file 
                                                          [(keyword "usernames-assoc")] 
                                                          (merge ((keyword "usernames-assoc")
                                                                   username-file)
                                                                  {(keyword username) id})))))
    (swap! search-ai-username-list (fn [v] (vec (concat v [username]))))
    (swap! search-ai-username-assoc (fn [m] (merge m {(keyword username) id})))))

(defn search-ai-check-username [username] (belongs username @search-ai-username-list))

(defn image-url-to-base64 [url]
  (let [response (client/get url {:as :byte-array})
        image-bytes (:body response)]
    (.encodeToString (Base64/getEncoder) image-bytes)))

(defn search-ai [user-name user-id mmap query response img]
  (m/trigger-typing-indicator! (:rest @state)
                               (:channel_id mmap))
  (if (not (search-ai-check-username user-name))
    (search-ai-add-username user-name user-id))

  (println img)
  (println (:content-type img))
  (println (str (:content-type img)))
  (println (and img (re-matches #"image\/.+" (str (:content-type img)))))
  (if (and img (re-matches #"image\/.+" (str (:content-type img))))
    (let [url "https://openrouter.ai/api/v1/chat/completions"
          headers {"Content-Type" "application/json"
                   "Authorization" (str "Bearer " (:openrouter config))}
          messages (swap! messages-ctx (fn [v] (vec (concat v
                                                           [{:role user-name
                                                             :content [{:type "text" :text (or (str query) "What do you think about this image?")}
                                                                       {:type "image_url" :image_url (str "data:image/jpeg;base64," (image-url-to-base64 (str (:url img))))}]}]))))
          body (generate-string
                {:model (nth image-model-list (first (rest @model-idx)))
                 :messages messages})
          print-this-stuff
          (println {:headers headers
                    :body body
                    :content-type :json})
          reply-json
          (parse-string
           (:body
            (client/post url
                         {:headers headers
                          :body body
                          :throw-exceptions false
                          :content-type :json}))
           true)
          reply-message-unprocessed
          (:message
           (nth
            (:choices reply-json) 0))
          print-this-stuff-again
          (println reply-message-unprocessed)
          reply-message
          (if reply-message-unprocessed
            {:role (:role reply-message-unprocessed)
             :content
             (reduce (fn [res ss] (s/replace res ss (mention-user ((keyword ss) @search-ai-username-assoc))))
                     (:content reply-message-unprocessed)
                     @search-ai-username-list)}
            reply-message-unprocessed)]
      (println reply-json)
      (println reply-message-unprocessed)
      (println reply-message)
      (if (not (not reply-message))
        (do
          (swap! messages-ctx (fn [v] (vec (concat v
                                                  [reply-message]))))
          (reply mmap (:content reply-message)))
        (do
          (swap! messages-ctx (fn [v] (vec (butlast v))))
          (swap! model-idx (fn [mdidx]
                             [(jt/plus (jt/local-date-time)
                                       (jt/days 1))
                              (mod (inc (first (rest mdidx))) 5)]))
          (search-ai user-name user-id mmap query response img))))
    
    (let [url "https://openrouter.ai/api/v1/chat/completions"
          headers {"Content-Type" "application/json"
                   "Authorization" (str "Bearer " (:openrouter config))}
          messages (swap! messages-ctx (fn [v] (vec (concat v
                                                           [{:role user-name
                                                             :content (str query)}]))))
          body (generate-string
                {:model (nth model-list (first (rest @model-idx)))
                 :messages messages})
          print-this-stuff
          (println {:headers headers
                    :body body
                    :content-type :json})
          reply-json
          (parse-string
           (:body
            (client/post url
                         {:headers headers
                          :body body
                          :throw-exceptions false
                          :content-type :json}))
           true)
          reply-message-unprocessed
          (:message
           (nth
            (:choices reply-json) 0))
          print-this-stuff-again
          (println reply-message-unprocessed)
          reply-message
          (if reply-message-unprocessed
            {:role (:role reply-message-unprocessed)
             :content
             (reduce (fn [res ss] (s/replace res ss (mention-user ((keyword ss) @search-ai-username-assoc))))
                     (:content reply-message-unprocessed)
                     @search-ai-username-list)}
            reply-message-unprocessed)]
      (println reply-json)
      (println reply-message-unprocessed)
      (println reply-message)
      (if (not (not reply-message))
        (do
          (swap! messages-ctx (fn [v] (vec (concat v
                                                  [reply-message]))))
          (reply mmap (:content reply-message)))
        (do
          (swap! messages-ctx (fn [v] (vec (butlast v))))
          (swap! model-idx (fn [mdidx]
                             [(jt/plus (jt/local-date-time)
                                       (jt/days 1))
                              (mod (inc (first (rest mdidx))) 5)]))
          (search-ai user-name user-id mmap query response img))))))
      
(def search-engine "https://search.zeroish.xyz/api.php?q=")
(def fallback-search-engine "https://searx.tuxcloud.net/search?q=")

(defn search [query]
  (or (:body (client/get (str search-engine query) {:as :json}))
      (:body (client/get (str fallback-search-engine query) {:as :json}))))

(defn search-image [query]
  (or (:body (client/get (str search-engine query "&category_images=") {:as :json}))
      (:body (client/get (str fallback-search-engine query "&t=1") {:as :json}))))

(defn search-yt [query]
  (:body (client/get (str "https://inv.tux.pizza/api/v1/search?q=" query "&pretty=1&fields=videoId,title") {:as :json})))

(defn yui-image-search [ch-id query]
  (let [result (search-image query)]
    (if (:thumbnail (car result))
      (prompt ch-id (str (:thumbnail (car result))))
      (prompt ch-id "No results found!"))))

(defn yui-yt-search [ch-id query]
  (let [result (search-yt query)]
    (if (:title (car result))
      (and
        (prompt ch-id (str "**Here's the video you requested:**\n" (:title (car result))))
        (prompt ch-id (str "https://youtu.be/" (:videoId (car result)))))
      (prompt ch-id "No results found!"))))

(defn yui-search [ch-id query]
  (let [result (search query)]
    (if (car result)
      (if (:title (car result))
        (prompt ch-id (str "**" (:title (car result)) "**\n"
                           (:description (car result)) "\n"
                           "\nFrom: " (unspace (:url (car result)))))
        (prompt ch-id (str (:response (:special_response (car result))) "\n"
                           "\nFrom: " (unspace (:source (:special_response (car result)))))))
      (prompt ch-id "No results found!"))))

(defn man-page [mmap cmd]
  (let [qqq (:out (shell/sh "curl" "-s" (str "https://cht.sh/" (car cmd) "?qT&style=bw")))]
    (reply mmap (subs qqq (s/index-of qqq "\n") (min (count qqq) 2000)))))

;; reminders

;; time.edn contains all our reminders in a map
;; all reminders are sorted
;; we have a function running every 10 minutes checking if any is in 10min range
;; if it is, it creates an async fun that triggers a message when it's reached
;; the bot then replies to the message creating the timer

(def time-file "~/.yui/time.edn")

(def yui-time-format (jtf/formatter "dd/MM/yyyy-HH:mm:ss"))

(defn call-reminder [reminder]
  (go
    (println "Reminder Spawned.")
    (let [secs (max 0
                    (jt/as (jt/duration
                             (jt/local-date-time)
                             (jt/local-date-time "dd/MM/yyyy-HH:mm:ss" (car reminder))) :seconds))]
      (println (format "Seconds left: %s" secs))
      (<! (timeout (* secs 1000)))
      (reply (car (cdr reminder))
             (car (cdr (cdr reminder)))))))

(defn ten-minute-check []
  (let [next-10-min (jt/plus (jt/local-date-time)
                             (jt/minutes 10))]
    (println "Doing a ten-minute-check.")
    (loop [remind []
           timings (edn/read-string (slurp time-file))]
      (if (car remind) (call-reminder (car remind)))
      (if (jt/before? (first @model-idx) ; (jt/local-date-time "dd/MM/yyyy-HH:mm:ss" (first @ctx))
                      (jt/local-date-time))
        (and (reset! model-idx [(jt/plus (jt/local-date-time) (jt/days 1))
                                0])
             (println "Reset model-idx!")))
      (if (and (car timings)
               (jt/before? (jt/local-date-time
                            "dd/MM/yyyy-HH:mm:ss" (car (car timings))) ; [(<time> messg reply-to ...) ...]
                           (jt/local-date-time next-10-min)))
        (recur (conj remind (car timings))
               (cdr timings))
        (binding [*print-length* -1]
          (prn-str (spit time-file
                         (or timings []))))))))

(defn parse-duration [time-str]
  ;; regex match
  (loop [time (jt/local-date-time)
         ent (s/split time-str #",")]
    (println (format "Parsing Duration %s" (car ent)))
    (if (and (car ent)
             (re-matches #"(\w+)([hdms])" (car ent)))
      (let [DUR ((fn [x] (cons (Integer/parseInt (car (cdr x)))
                              (rest (rest x))))
                 (re-matches #"(\w+)([hdms])" (car ent)))]
        (case (car (cdr DUR))
          "d"
          (recur (jt/plus time (jt/days (car DUR)))
                 (cdr ent))
          "h"
          (recur (jt/plus time (jt/hours (car DUR)))
                 (cdr ent))
          "m"
          (recur (jt/plus time (jt/minutes (car DUR)))
                 (cdr ent))
          "s"
          (recur (jt/plus time (jt/seconds (car DUR)))
                 (cdr ent))
          "ERROR"))
      time)))

(defn parse-date [time-str]
  (println "Parsing Date")
  (if (re-find #"-" time-str)
    ;; full regex
    (let [time-arr (re-matches #"(\d+)/(\d+)/?(\d+)?-(\d+):(\d+):?(\d+)?" time-str)
          day (nth time-arr 1)
          month (nth time-arr 2)
          year (nth time-arr 3)
          hour (nth time-arr 4)
          minutes (nth time-arr 5)
          seconds (nth time-arr 6)]
      (jt/local-date-time (if year (Integer/parseInt year) 2024)
                          (Integer/parseInt month)
                          (Integer/parseInt day)
                          (Integer/parseIntger/parseInt hour)
                          (Integer/parseIntger/parseInt minutes)
                          (if seconds (Integer/parseInt seconds) 0)))
    (if (re-find #"/" time-str)
      ;; date regex
      (let [time-arr (re-matches #"(\d+)/(\d+)/?(\d+)?" time-str)
            day (nth time-arr 1)
            month (nth time-arr 2)
            year (nth time-arr 3)]
        (jt/local-date-time (if year (Integer/parseInt year) 2024)
                            (Integer/parseInt month)
                            (Integer/parseInt day)))
      ;; time regex
      (let [time-arr (re-matches #"(\d+):(\d+):?(\d+)?" time-str)
            hour (nth time-arr 1)
            minutes (nth time-arr 2)
            seconds (nth time-arr 3)]
        (jt/local-date-time (jt/as (jt/local-date-time) :year)
                            (jt/as (jt/local-date-time) :month-of-year)
                            (jt/as (jt/local-date-time) :day-of-month)
                            (Integer/parseIntger/parseInt hour)
                            (Integer/parseIntger/parseInt minutes)
                            (if seconds (Integer/parseInt seconds) 0))))))

(defn remind [in-or-at time-str text mmap]
  (println "Triggered Reminder fn.")
  (let [time (case in-or-at
               in (parse-duration time-str)
               at (parse-date time-str))]
    (println "Finished Parsing Time String.")
    (if (= time "ERROR")
      (reply mmap "Bad reminder formatting. Use dd/mm[/yy][-]HH:mm[:ss] or n[d][h][m][s][,]")
      (do
        (println "Reading time-file...")
        (reply mmap (str "Set reminder for " (jtf/format yui-time-format time))) 
        (let [timings (edn/read-string (slurp time-file))]
          (println "Read time-file.")
          (binding [*print-length* -1]
            (prn-str (spit time-file
                           (sort
                             #(jt/duration
                                (jt/local-date-time "dd/MM/yyyy-HH:mm:ss" (car %))
                                (jt/local-date-time "dd/MM/yyyy-HH:mm:ss" (car %)))
                             (conj timings
                                   [(jtf/format yui-time-format time) ; exact time
                                    mmap ; replying to
                                    text])))))) ; message
        (ten-minute-check)))))

;; eval code

(defn live-repl [ch-id content]
  (let [code (detokenize (remove (fn [x] (and (< 2 (count x))
                                              (= "```" (subs x 0 3))))
                                 content))]
    (prompt ch-id (str (eval (read-string code))))))

;; dice roll
(defn roll 
  ([ch-id]
   (prompt ch-id (str ":game_die: You rolled a " (inc (rand-int 6)) "! :game_die:"))
   (prompt ch-id "https://tenor.com/view/girl-waiting-anime-chill-rolling-gif-15974128"))
  ([ch-id number]
   (prompt ch-id (str ":game_die: You rolled a " (inc (rand-int number)) "! :game_die:"))
   (prompt ch-id "https://tenor.com/view/girl-waiting-anime-chill-rolling-gif-15974128"))
  ([ch-id die number]
   (let [die-roll-values (repeatedly die #(inc (rand-int number)))]
     (prompt ch-id (str ":game_die: You rolled a " (reduce #'+ die-roll-values) "! :game_die:"))
     (prompt ch-id (str "individual values are " (pr-str die-roll-values)))
     (prompt ch-id "https://tenor.com/view/girl-waiting-anime-chill-rolling-gif-15974128"))))

(defn pain [ch-id]
  (prompt ch-id "https://tenor.com/view/k-on-yui-hirasawa-pain-gif-23894830"))

(defn random-gif [mmap]
  (let [gif-file (edn/read-string (slurp "~/.yui/gifs.edn"))]
    (reply mmap (rand-nth (:gifs gif-file)))))

(defn add-gif [mmap file]
  (let [gif-file (edn/read-string (slurp "~/.yui/gifs.edn"))]
    (if (not (belongs (:url file) (:gifs gif-file)))
      (and
       (spit "~/.yui/gifs.edn"
             (binding [*print-length* -1] (prn-str (assoc-in gif-file 
                                                             [(keyword "gifs")] 
                                                             (concat ((keyword "gifs") gif-file)
                                                                     [(:url file)])))))
       (reply mmap "Added file to collection!"))
      (reply mmap "That file already exists in the collection! BAKA!!"))))

(defn predict [ch-id]
  (prompt ch-id (str "My calculations say the chances are **" (rand-int 101) "%**.")))

(defn say-x [ch-id text]
  (prompt ch-id text))

(defn add-todo [ch-id text]
  (prompt (:todo-channel config) text)
  (prompt ch-id (str \" text \" " was added to TODO!")))

(defn say-hi []
  (str (rand-nth (:say-hi config))))

(defn say-bye [ch-id user]
  (prompt ch-id (str (rand-nth (:say-bye config)) ", " (mention-user user) \!)))

(defn goodbye [ch-id]
  (prompt ch-id "Bye!")
  (prompt ch-id "https://tenor.com/view/yui-x-azusa-yui-azusa-azunyan-anime-gif-21336402")
  (c/disconnect-bot! (:gateway @state)))

(defn say-error [ch-id com]
  (prompt ch-id (str "I don't know how to " com ", baka!")))

(defn associated-key [user]
  (let [keyword-file (edn/read-string (slurp "keys.edn"))]
    ((keyword (str (:id user))) keyword-file)))

(defn caption [ch-id file ctext]
  (clojure.java.io/copy
    (:body (client/get (str (:url file)) {:as :stream}))
    (java.io.File. "/tmp/img"))
  (prompt ch-id "Processing...")
  (let [width (Integer/parseInt (:out (apply shell/sh (tokenize "identify -format %w /tmp/img"))))
        fmt (s/lower-case (:out (apply shell/sh (tokenize "identify -format %m /tmp/img"))))]
    (let [answer (:out (apply shell/sh (concat (tokenize (str "convert /tmp/img -background none -font Upright -fill white -stroke black -strokewidth " (int (/ width 200)) " -size " width "x" (int (/ width 2)) " -gravity center"))
                                               (list (str "caption:" ctext))
                                               (tokenize (str "-composite /tmp/img." fmt)))))]
      (prompt ch-id "Processing done!"))
    (m/create-message! (:rest @state) ch-id :file (java.io.File. (str "/tmp/img." fmt)))))

(defn subscribe [mmap author msg]
  (let [sub-file (edn/read-string (slurp "~/.yui/subs.edn"))]
    ;; yui sub create name
    ;; -> creates sub group
    (if (= (car msg) "create")
      (if (not (car (cdr msg)))
        (reply mmap "Enter a name for the new group!")
      (let [group-name (apply str (re-seq #"\w" (car (cdr msg))))]
        (if ((keyword group-name) sub-file)
          (reply mmap "A sub group with the same name already exists!")
          (do
            (spit "~/.yui/subs.edn"
                  (binding [*print-length* -1] (prn-str (assoc-in sub-file 
                                                                  [(keyword group-name)]
                                                                  (list (:id author))))))
            (reply mmap "Sub group created!")))))
      ;; yui sub name
      ;; -> joins sub group
      (let [group-name (apply str (re-seq #"\w" (car msg)))]
        (spit "~/.yui/subs.edn"
              (binding [*print-length* -1] (prn-str (assoc-in sub-file 
                                                              [(keyword group-name)] 
                                                              (concat ((keyword group-name) sub-file)
                                                                      [(:id author)])))))
        (reply mmap (str "Joined the sub group " group-name))))))

(defn unsubscribe [ch-id msg])

(defn sub-ping [mmap msg]
  (let [sub-file (edn/read-string (slurp "~/.yui/subs.edn"))]
    ;; yui ping name
    ;; pings sub group
    (let [group-name (apply str (re-seq #"\w" msg))]
      (if (not ((keyword group-name) sub-file))
        (reply mmap "Sub group does not exist!")
        (reply mmap (detokenize (mapv mention-user ((keyword group-name) sub-file))))))))


(defn image-name [img]
  (let [image-file (edn/read-string (slurp "~/.yui/images.edn"))]
    ((keyword img) image-file)))

(defn image-listing [mmap]
  (let [image-file (edn/read-string (slurp "~/.yui/images.edn"))]
    (reply mmap (s/join ", " (map (fn [str] (s/replace str #"-" " "))
                                    (cdr (s/split (apply str (keys image-file)) #":")))))))

(defn counter-add [ch-id f-author]
  (let [author (:id f-author)
        score (edn/read-string (slurp "~/.yui/score.edn"))]
    (if (not ((keyword author) score))
      (spit "~/.yui/score.edn"
            (binding [*print-length* -1] (prn-str (assoc-in score
                                                            [(keyword author)] 
                                                            1))))
      (let* [count (inc ((keyword author) score))
             new-score (assoc-in score
                                 [(keyword author)] 
                                 count)]
        (spit "~/.yui/score.edn"
              (binding [*print-length* -1] (prn-str new-score)))))))

(defn leaderboard [ch-id gd-id]
  (def number (atom 0))
  (let [score (edn/read-string (slurp "~/.yui/score.edn"))]
    (prompt ch-id
            (str "# POGGERS LEADERBOARD\n"
                 (apply str
                        (map (fn [[user-id val]]
                               (let* [user (m/get-user! (:rest @state)
                                                        (Long/parseLong (subs (str user-id) 1)))
                                      name (:username @user)]
                                 (if (not (s/blank? name))
                                   (str (swap! number inc) ". " name ": " val "\n"))))
                             (into (sorted-map-by
                                    (fn [key1 key2]
                                      (<= (key2 score)
                                          (key1 score))))
                                   score)))))))

(defn add-image [mmap key-name file]
  (let [image-file (edn/read-string (slurp "~/.yui/images.edn"))]
    (if (not ((keyword key-name) image-file))
      (let* [stamp (str (quot (System/currentTimeMillis) 1000))
             image (clojure.java.io/copy
                     (:body (client/get (str (:url file)) {:as :stream}))
                     (java.io.File. (str "~/.yui/file_" stamp)))
             fmt (s/lower-case (s/trim-newline (car (s/split (:out (apply shell/sh (tokenize (str "file -b --extension ~/.yui/file_" stamp)))) #"/"))))
             final-image (:out (apply shell/sh 
                                      (tokenize 
                                        (str "mv ~/.yui/file_" stamp 
                                             " ~/.yui/file_" stamp "." fmt))))]
        (spit "~/.yui/images.edn"
              (binding [*print-length* -1] (prn-str (merge image-file 
                                                           {(keyword key-name) 
                                                            (str "file_" stamp "." fmt)}))))
        (reply mmap "Keyword registered with the given file!"))
      (reply mmap "Keyword already exists!"))))

(defn update-image [mmap key-name file]
  (let [image-file (edn/read-string (slurp "~/.yui/images.edn"))]
    (if ((keyword key-name) image-file)
      (let* [stamp (str (quot (System/currentTimeMillis) 1000))
             image (clojure.java.io/copy
                     (:body (client/get (str (:url file)) {:as :stream}))
                     (java.io.File. (str "~/.yui/file_" stamp)))
             fmt (s/lower-case (s/trim-newline (car (s/split (:out (apply shell/sh (tokenize (str "file -b --extension ~/.yui/file_" stamp)))) #"/"))))
             final-image (:out (apply shell/sh 
                                      (tokenize 
                                        (str "mv ~/.yui/file_" stamp 
                                             " ~/.yui/file_" stamp "." fmt))))]
        (spit "~/.yui/images.edn"
              (binding [*print-length* -1] (prn-str (merge image-file 
                                                           {(keyword key-name) 
                                                            (str "file_" stamp "." fmt)}))))
        (reply mmap "Keyword updated with the given file!"))
      (reply mmap "Keyword does not exist!"))))

(defn yui-show [mmap img]
  (m/create-message! (:rest @state)
                     (:channel_id mmap)
                     :message-reference mmap
                     :file (java.io.File. (str "~/.yui/" (image-name img)))))

(defn pin-message [ch-id ref-msg] ;; broken
  (if ref-msg
    (do
      (m/add-channel-pinned-message! (:rest @state) ch-id ref-msg)
      (prompt ch-id "Pinned the message!"))
    (prompt ch-id "No message mentioned, baka!")))

(defn delete-message [ch-id ref-msg]
  (if ref-msg
    (do
      (m/delete-message! (:rest @state) ch-id ref-msg)
      (prompt ch-id "Deleted the message!"))
    (prompt ch-id "No message mentioned, baka!")))

(defn edit-call [ch-id msg-id]
  (m/edit-message! (:rest @state)
                   ch-id
                   msg-id
                   :content
                   "Edited by Yui <3"))

(defn not-a-mod [ch-id]
  (prompt ch-id "You are not a mod, you sussy baka!"))

(defn say-help [ch-id]
  (prompt ch-id (:help-string config)))

(defn kill-person [ch-id user]
  (prompt ch-id (str "_" (rand-nth (:kill-methods config)) " " (:username user) "_")))

(defn dm-privacy [ch-id user]
  (prompt (m/create-dm! (:rest @state) (:id user))
          (:privacy-string config)))

(defn command-audio [ch-id com]
  (case com
    "play" (prompt ch-id "nothingburger")))

