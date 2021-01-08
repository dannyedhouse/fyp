import { Component, OnInit } from '@angular/core';
import { FormGroup, FormBuilder, Validators,ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  inputForm: FormGroup;
  submitted : boolean;

  constructor(private fb: FormBuilder, private router: Router) {
    this.createForm();
  }

  ngOnInit() {}

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
      this.router.navigate(['/summary'], { queryParams: {url: url}})
    }
  }
}
