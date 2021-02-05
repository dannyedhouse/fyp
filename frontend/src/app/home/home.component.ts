import { Component, OnInit } from '@angular/core';
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../api.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  inputForm: FormGroup;
  submitted : boolean;
  errorMessage = "";
  loading: boolean;

  constructor(private fb: FormBuilder, private router: Router, private apiService: ApiService) {
    this.createForm();
  }

  ngOnInit() {
    this.loading = false;
  }

  createForm() {
    const regex = /^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+$/;
    this.inputForm = this.fb.group({
      url:['', [Validators.required, Validators.pattern(regex)]],
    });
  }

  onSubmit() {
    this.submitted = true;
    if(this.inputForm.valid) {
      var url = this.inputForm.get('url').value;
      this.loading = true;
      this.apiService.getSummary(url).subscribe((value: any)=>{
        this.loading = false;
        if (Object.keys(value).length == 0) {
          this.displayAlert("No results found. Please try a different URL");
        } else {
          this.router.navigate(['/summary'], { queryParams: {url: url}})
        }    
      });
    }
  }

  displayAlert(error: string): void {
    this.errorMessage = error;
    setTimeout(() => {
      this.closeAlert();
    }, 3000);
  }

  closeAlert(): void {
    this.errorMessage = "";
  }
}
