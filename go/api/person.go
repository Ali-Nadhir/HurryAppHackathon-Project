package api

type Person struct {
	ID      int    `json:"id"`
	Name    string `json:"name"`
	Email   string `json:"email"`
	Phone   string `json:"phone"`
	Role    string `json:"role"`
	Address string `json:"address"`
	Age     int    `json:"age"`
}

type Fingerprint struct {
	ID       int `json:"id"`
	PersonID int `json:"person_id"`
	Finger   int `json:"finger"`
}
